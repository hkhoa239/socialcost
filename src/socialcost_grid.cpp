// C entry points over arena_social_cost_layer's cost functions, for offline labels.
//
// The cost terms come straight from arena_social_cost_layer/cost_math.hpp. The
// compositing below mirrors SocialCostLayer::updateCosts (social_cost_layer.cpp):
// every cell centre within personal_space_radius of a pedestrian takes the
// personal-space term, every cell centre within group_extra_radius of the
// segment between two members of a weighted interaction takes the group term,
// each term is mapped through toCellCost, and a cell keeps the maximum.
// tests/test_live_agreement.py checks the result against the running plugin.

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "arena_social_cost_layer/cost_math.hpp"

namespace
{

// arena_people_msgs/Pedestrian animation states, as switched on in the layer.
constexpr uint8_t kPanic = 3;
constexpr uint8_t kSurprised = 4;
constexpr uint8_t kCurious = 5;
constexpr uint8_t kThreatening = 6;

}  // namespace

extern "C" {

struct ScParams
{
  double personal_space_radius;
  double personal_space_sigma;
  double group_extra_radius;
  double group_sigma;
  double falloff_shape;
  int max_cost_value;  // already clamped to [0, 252] by the caller, as the layer does
  double cost_scale;
  double cost_bias;
};

struct ScPed
{
  double x;
  double y;
  uint8_t animation_state;
  int32_t interaction_id;
  uint8_t interaction_type;
};

/// Paints the social layer alone into out[j * width + i] (uint8, zero-initialised here).
///
/// Cell (i, j) has its centre at (ox + (i + 0.5) res, oy + (j + 0.5) res) in the grid
/// frame, and the grid frame sits at (gx, gy, gyaw) in the pedestrians' frame. With
/// gyaw = 0 and (gx, gy) = 0 this is exactly a Costmap2D with origin (ox, oy).
/// weights[k] is interaction_weights[k], n_weights its length.
int sc_paint(
  const ScParams * p, const ScPed * peds, int n_peds, const double * weights, int n_weights,
  double gx, double gy, double gyaw, double ox, double oy, double res, int width, int height,
  uint8_t * out)
{
  using arena_social_cost_layer::groupBetweenCost;
  using arena_social_cost_layer::personalSpaceCost;
  using arena_social_cost_layer::PersonalSpaceParams;
  using arena_social_cost_layer::pointToSegmentDistSq;

  if (width <= 0 || height <= 0 || res <= 0.0) {
    return -1;
  }
  std::fill(out, out + static_cast<size_t>(width) * height, uint8_t{0});

  // SocialCostLayer::onInitialize derives alert/danger sigmas by scaling the defaults.
  const PersonalSpaceParams defaults{};
  const double sigma_scale = p->personal_space_sigma / defaults.base_sigma;
  const double base_sigma = p->personal_space_sigma;
  const double alert_sigma = defaults.alert_sigma * sigma_scale;
  const double danger_sigma = defaults.danger_sigma * sigma_scale;

  auto to_cell = [p](double normalized) -> uint8_t {
      const double scaled = std::clamp(normalized * p->cost_scale + p->cost_bias, 0.0, 1.0);
      return static_cast<uint8_t>(scaled * p->max_cost_value);
    };
  auto weight_of = [weights, n_weights](uint8_t type) -> double {
      return type < n_weights ? weights[type] : 0.0;
    };

  const double c = std::cos(gyaw);
  const double s = std::sin(gyaw);
  const double clip_p = p->personal_space_radius * p->personal_space_radius;
  const double clip_g = p->group_extra_radius * p->group_extra_radius;

  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      const double lx = ox + (i + 0.5) * res;
      const double ly = oy + (j + 0.5) * res;
      const double wx = (gyaw == 0.0) ? gx + lx : gx + c * lx - s * ly;
      const double wy = (gyaw == 0.0) ? gy + ly : gy + s * lx + c * ly;
      uint8_t best = 0;

      for (int k = 0; k < n_peds; ++k) {
        const ScPed & ped = peds[k];
        const double dx = wx - ped.x;
        const double dy = wy - ped.y;
        if (dx * dx + dy * dy > clip_p) {
          continue;
        }
        double sigma = base_sigma;
        double amplitude = defaults.base_amplitude;
        if (ped.animation_state == kPanic || ped.animation_state == kThreatening) {
          sigma = danger_sigma;
          amplitude = defaults.danger_amplitude;
        } else if (ped.animation_state == kSurprised || ped.animation_state == kCurious) {
          sigma = alert_sigma;
          amplitude = defaults.alert_amplitude;
        }
        best = std::max(best, to_cell(personalSpaceCost(dx, dy, sigma, amplitude, p->falloff_shape)));
      }

      // Pairs inside one interaction_id, in roster order, as interactionGroups() builds them.
      for (int a = 0; a < n_peds; ++a) {
        if (peds[a].interaction_id < 0) {
          continue;
        }
        bool first_of_group = true;
        for (int q = 0; q < a; ++q) {
          if (peds[q].interaction_id == peds[a].interaction_id) {
            first_of_group = false;
            break;
          }
        }
        if (!first_of_group) {
          continue;
        }
        // The group takes the interaction_type of its first member, as in the layer.
        const double weight = weight_of(peds[a].interaction_type);
        if (weight <= 0.0) {
          continue;
        }
        for (int m = a; m < n_peds; ++m) {
          if (peds[m].interaction_id != peds[a].interaction_id) {
            continue;
          }
          for (int n = m + 1; n < n_peds; ++n) {
            if (peds[n].interaction_id != peds[a].interaction_id) {
              continue;
            }
            const ScPed & pa = peds[m];
            const ScPed & pb = peds[n];
            if (pointToSegmentDistSq(wx, wy, pa.x, pa.y, pb.x, pb.y) > clip_g) {
              continue;
            }
            best = std::max(
              best, to_cell(
                groupBetweenCost(
                  wx, wy, pa.x, pa.y, pb.x, pb.y, weight, p->group_sigma,
                  p->falloff_shape)));
          }
        }
      }
      out[static_cast<size_t>(j) * width + i] = best;
    }
  }
  return 0;
}

}  // extern "C"
