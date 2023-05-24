#ifndef DUMMYSTATS_H
#define DUMMYSTATS_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <map>
#include <mutex>


namespace mlir
{
  class Pass;

  namespace dummyStats
  {
    std::unique_ptr<mlir::Pass> createDummyStatsPass();
    class datum{
      public:
        datum() : occurances(0) {}
        ~datum() {}
        datum& operator=(const datum& other);
        int incOccurances()   {lcl.lock(); this->occurances++; lcl.unlock(); return 0;}
        int getOccurances()   {lcl.lock(); return this->occurances; lcl.unlock(); return 0;}
        int resetOccurances() {lcl.lock(); this->occurances=0; lcl.unlock(); return 0;}
      private:
        long int occurances;
        std::mutex lcl;

    };
    extern std::map<std::string, datum> statsMap;


  } // end namespace dummyStatsPass
} // end namespace mlir

#endif /*dummyStats.h*/