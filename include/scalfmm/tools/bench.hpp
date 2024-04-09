// --------------------------------
// See LICENCE file at project root
// File : algorithm/utils/bench.hpp
// --------------------------------
#pragma once

#include <cpp_tools/timers/simple_timer.hpp>
#include <stdlib.h>
#include <cctype>
#include <algorithm>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <tuple>

namespace scalfmm::bench
{
  template<typename Duration>
  inline auto compute(std::unordered_map<std::string,cpp_tools::timers::timer<Duration>> timers)
  {
    auto overall = Duration::zero().count();
    
    for(auto e : timers)
    {
      overall += e.second.elapsed();
    }
    auto neartime = timers["p2p"].elapsed();
    auto fartime = overall - neartime;
    auto ratio{double(fartime)/double(neartime)};
  
    return std::make_tuple(fartime, neartime, overall, ratio);
  }

  template<typename Duration>
  inline auto print(std::unordered_map<std::string,cpp_tools::timers::timer<Duration>> timers)
  {
    auto [fartime, neartime, overall,ratio] = compute(timers);

    std::cout << "[time][bottom pass]       : " << timers["p2m"].elapsed() << "\n";
    std::cout << "[time][upward pass]       : " << timers["m2m"].elapsed() << "\n";
    std::cout << "[time][transfer pass]     : " << timers["m2l"].elapsed() << "\n";
    std::cout << "[time][downward pass]     : " << timers["l2l"].elapsed() << "\n";
    std::cout << "[time][cell_to_leaf pass] : " << timers["l2p"].elapsed() << "\n";
    std::cout << "[time][direct pass]       : " << timers["p2p"].elapsed() << "\n";
    std::cout << "[time][far time]   : " << fartime << '\n';
    std::cout << "[time][near time]  : " << neartime << '\n';
    std::cout << "[time][diff time]  : " << std::abs(fartime - neartime) << '\n';
    std::cout << "[time][ratio time] : " << ratio << '\n';
    std::cout << "[time][full algo]  : " << overall << '\n';
    return std::make_tuple(fartime, neartime, overall, ratio);
  }
  
  template<typename String, typename... Strings>
  inline auto dump_csv(std::string file_name, std::string header, String arg, Strings... args)
  -> void
  {
    std::ofstream benchfile;
    if(std::filesystem::exists(file_name))
    {
      benchfile.open(file_name, std::ios::app);
    }
    else
    {
      benchfile.open(file_name);
      benchfile << header << '\n';
    }
    benchfile << arg;
    (benchfile << ... << (','+args)); 
    benchfile << '\n';
    benchfile.close();
  }
}
