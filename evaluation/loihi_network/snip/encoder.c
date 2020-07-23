/*
Encoder SNIP: Encode input spike activity and Bias spikes
*/
#include <stdlib.h>
#include "nxsdk.h"
#include "encoder.h"
#define input_num 24
#define encode_num 6
#define bias_num 4

int epoch = 5 + 5; // Base epoch time 5 can be changed to other spiking timesteps (10, 25, 50)

int en_spike_prob[input_num] = {0};

int en_spike_prob_precision = 100;

int en_input_core = 0;
int en_input_chip = 0;
int en_input_start = 0;

int en_bias_core = 1;
int en_bias_chip = 0;
int en_bias_start = 0;

int en_decode_num_bits = 7;
int en_decode_overall_bits = 28;
int en_decode_per_int_num = 4;

int do_encoder(runState *s)
{
    return 1;
}

void run_encoder(runState *s)
{
    int time = s->time_step;
    if(time % epoch == 1)
    {
        int InputChannelId = getChannelID("encodeinput");
        int tmp_input[input_num];
        readChannel(InputChannelId, &tmp_input, encode_num);
        /*
        for(int i=0; i<input_num; i++)
        {
            en_spike_prob[i] = tmp_input[i];
        }
        */
        int decode_output_idx = 0;
        for(int i=0; i<encode_num; i++)
        {

            int big_int = tmp_input[i];
            for(int j=0; j<en_decode_per_int_num; j++)
            {
                en_spike_prob[decode_output_idx] = big_int - ((big_int >> en_decode_num_bits) << en_decode_num_bits);
                big_int = big_int >> en_decode_num_bits;
                decode_output_idx++;
            }
        }
    }
    /*
    Generate Poisson Spikes and Send to Input Neurons
    */
    if(time % epoch >= 1 && time % epoch <= (epoch - 5))
    {
        for(int i=0; i<input_num; i++)
        {
            int random_num = rand();
            int spike_threshold = (RAND_MAX / en_spike_prob_precision) * en_spike_prob[i];
            if(random_num < spike_threshold)
            {
                int input_axon_id = en_input_start + i;
                uint16_t axonId = 1<<14 | ((input_axon_id) & 0x3FFF);
                ChipId chipId = nx_nth_chipid(en_input_chip);
                nx_send_remote_event(time, chipId, (CoreId){.id=4+en_input_core}, axonId);
            }
        }
    }
    /*
    Send Spikes to Bias Neurons at each layer (3 Hidden layers + Output layer)
    */
    for(int i=0; i<bias_num; i++)
    {
        if(time % epoch > i && time % epoch <= i + 5)
        {
            int bias_axon_id = en_bias_start + i;
            uint16_t axonId = 1<<14 | ((bias_axon_id) & 0x3FFF);
            ChipId chipId = nx_nth_chipid(en_bias_chip);
            nx_send_remote_event(time, chipId, (CoreId){.id=4+en_bias_core}, axonId);
        }
    }
}
