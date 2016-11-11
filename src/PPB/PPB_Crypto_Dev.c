#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/dev/ppb_crypto_dev.h>

#include "log.h"

#include <stdlib.h>
#include <time.h>

void GetRandomBytes(char* buffer, uint32_t num_bytes)
{
    uint32_t i;

    srand(time(NULL));

    for(i = 0; i < num_bytes; i++)
        buffer[i] = rand();
};

struct PPB_Crypto_Dev_0_1 PPB_Crypto_Dev_0_1_instance =
{
    .GetRandomBytes = GetRandomBytes,
};
