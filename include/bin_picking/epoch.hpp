#pragma once

#include <random>

#include <bin_picking/parameter.hpp>


struct Epoch {
  int number_cycles;
  SelectionMethod selection_method_primary;
  SelectionMethod selection_method_secondary {SelectionMethod::Random};
  double percentage_secondary;

  explicit Epoch() {}
  explicit Epoch(int number_cycles, const SelectionMethod& selection_method):
    number_cycles(number_cycles),
    selection_method_primary(selection_method),
    percentage_secondary(0.0) { }

  SelectionMethod getSelectionMethod() const {
    std::random_device random_device;
    std::mt19937 e2(random_device());
    std::uniform_real_distribution<> dist(0, 1);
    return (dist(e2) > percentage_secondary) ? selection_method_primary : selection_method_secondary;
  }

  SelectionMethod getSelectionMethodPerform(int count_failed_grasps_since_last_success) const {
    return (count_failed_grasps_since_last_success == 0) ? SelectionMethod::Max : SelectionMethod::Top5;
  }

  static bool selectionMethodShouldBeHigh(SelectionMethod method) {
    switch (method) {
      case SelectionMethod::Max:
      case SelectionMethod::Top5:
      case SelectionMethod::Uncertain:
      case SelectionMethod::NotZero: return true;
      default: return false;
    }
  }
};
