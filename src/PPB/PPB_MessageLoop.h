#ifndef PPB_MessageLoop_h
#define PPB_MessageLoop_h

extern pthread_t PPB_MessageLoop_main_thread;

extern int PPB_MessageLoop_push(PP_Resource message_loop,
    struct PP_CompletionCallback callback, int64_t delay_ms, int32_t result);

extern struct PPB_MessageLoop_1_0 PPB_MessageLoop_1_0_instance;

#endif /* PPB_MessageLoop_h */
