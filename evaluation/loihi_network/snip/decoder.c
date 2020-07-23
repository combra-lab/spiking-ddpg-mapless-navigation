/*
Decoder SNIP: Decode output neuron spikes and reset network every epoch
*/
#include <stdlib.h>
#include "nxsdk.h"
#include "decoder.h"
#define out_num 2

extern int epoch;

int de_out_spikes[out_num] = {0};

int de_hidden1_core_start = 3;
int de_hidden1_core_end = 4;

int de_hidden2_core_start = 5;
int de_hidden2_core_end = 6;

int de_hidden3_core_start = 7;
int de_hidden3_core_end = 8;

int de_output_core_start = 2;
int de_output_core_end = 2;

static int numNeuronsPerCore = 1024;
static int NUM_Y_TILES = 5;

int do_decoder(runState *s)
{
    return 1;
}

void run_decoder(runState *s)
{
    int time = s->time_step;
    /*
    Counting output spikes
    */
    if(time % epoch > 5 || time % epoch == 0)
    {
        for(int i=0; i<out_num; i++)
        {
            if(SPIKE_COUNT[(time)&3][i+0x20] > 0)
            {
                de_out_spikes[i] += 1;
            }
            SPIKE_COUNT[(time)&3][i+0x20] = 0;
        }
    }
    /*
    Send out spike counts
    */
    if(time % epoch == 0)
    {
        int OutputChannelId = getChannelID("decodeoutput");
        writeChannel(OutputChannelId, de_out_spikes, out_num);
        for(int i=0; i<out_num; i++)
        {
            de_out_spikes[i] = 0;
        }
    }
    /*
    Reset each layer separately at the end of operation
    */
    if(time & epoch == (epoch - 3) || time & epoch == (epoch - 2))
    {
        core_hard_reset(de_hidden1_core_start, de_hidden1_core_end);
    }
    if(time & epoch == (epoch - 2) || time & epoch == (epoch - 1))
    {
        core_hard_reset(de_hidden2_core_start, de_hidden2_core_end);
    }
    if(time & epoch == (epoch - 1) || time & epoch == 0)
    {
        core_hard_reset(de_hidden3_core_start, de_hidden3_core_end);
    }
    if(time & epoch == 0 || time & epoch == 1)
    {
        core_hard_reset(de_output_core_start, de_output_core_end);
    }
}

void core_hard_reset(int start, int end)
{
    NeuronCore *nc;
    CoreId coreId;

    CxState cxs = (CxState) {.U=0, .V=0};
    for(int i=start; i<end+1; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
        nx_fast_init64(nc->cx_state, numNeuronsPerCore, *(uint64_t*)&cxs);
    }

    for(int i=start; i<end+1; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
        nx_fast_init32(nc->dendrite_accum, 8192/2, 0);
    }

    MetaState ms = (MetaState) {.Phase0=2, .SomaOp0=3,
                                .Phase1=2, .SomaOp1=3,
                                .Phase2=2, .SomaOp2=3,
                                .Phase3=2, .SomaOp3=3};
    for(int i=start; i<end+1; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
        nx_fast_init32(nc->cx_meta_state, numNeuronsPerCore/4, *(uint32_t*)&ms);
    }
}

void logicalToPhysicalCoreId(int logicalId, CoreId *physicalId)
{
    physicalId->p = logicalId % 4;
    physicalId->x = logicalId/(4*(NUM_Y_TILES-1));
    physicalId->y = (logicalId - physicalId->x*4*(NUM_Y_TILES-1))/4 + 1;
}