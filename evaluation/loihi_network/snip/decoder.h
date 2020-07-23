#include "nxsdk.h"
int do_decoder(runState *s);
void run_decoder(runState *s);
void core_hard_reset(int start, int end);
void logicalToPhysicalCoreId(int logicalId, CoreId *physicalId);