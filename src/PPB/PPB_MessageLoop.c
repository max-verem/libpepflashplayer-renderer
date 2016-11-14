#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_errors.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_message_loop.h>

#include "log.h"
#include "res.h"

#include "PPB_MessageLoop.h"

static int64_t _now_us()
{
    int64_t r;
    struct timeval tv;

    gettimeofday(&tv, NULL);

    r = tv.tv_sec;
    r *= 1000000LL;
    r += tv.tv_usec;

    return r;
};

struct PPB_MessageLoop_1_0 PPB_MessageLoop_1_0_instance;

typedef struct message_loop_item_desc
{
    struct PP_CompletionCallback callback;
    int64_t delay_ms;
    int32_t result;
    int64_t now;
    struct message_loop_item_desc* next;
} message_loop_item_t;

typedef struct message_loop_desc
{
    PP_Instance instance_id;
    PP_Resource self;
    pthread_t thread;
    int registry_idx;
    message_loop_item_t* items;
    pthread_cond_t cond;
    int f_exit;
} message_loop_t;

typedef struct message_loop_map_desc
{
    PP_Instance msg_loop;
    pthread_t thread;
} message_loop_map_t;

static struct
{
    message_loop_map_t* list;
    pthread_mutex_t lock;
    int count;
} registry =
{
    .list = NULL,
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .count = 0
};

pthread_t PPB_MessageLoop_main_thread = 0;

static void Destructor(message_loop_t* ctx)
{
    message_loop_item_t* item;

    LOG("{%d}", ctx->self);

    /* free items */
    for(item = ctx->items; item;)
    {
        message_loop_item_t* tmp = item->next;
        free(item);
        item = tmp;
    };

    /* destroy conditional */
    pthread_cond_destroy(&ctx->cond);
};

/**
 * Creates a message loop resource.
 *
 * This may be called from any thread. After your thread starts but before
 * issuing any other PPAPI calls on it, you must associate it with a message
 * loop by calling AttachToCurrentThread.
 */
static PP_Resource Create(PP_Instance instance)
{
    int res = res_create(sizeof(message_loop_t), &PPB_MessageLoop_1_0_instance, (res_destructor_t)Destructor);

    message_loop_t* msg_loop = (message_loop_t*)res_private(res);

    msg_loop->instance_id = instance;
    msg_loop->self = res;
    pthread_cond_init(&msg_loop->cond, NULL);

    return res;
};


/**
 * Returns a resource identifying the message loop for the main thread. The
 * main thread always has a message loop created by the system.
 */
static PP_Resource GetForMainThread(void)
{
    int i;
    PP_Resource r = 0;

    pthread_mutex_lock(&registry.lock);

    if(PPB_MessageLoop_main_thread)
        for(i = 0; i < registry.count && !r; i++)
            if(PPB_MessageLoop_main_thread == registry.list[i].thread)
                r = registry.list[i].msg_loop;

    pthread_mutex_unlock(&registry.lock);

    return r;
};


/**
 * Returns a reference to the PPB_MessageLoop object attached to the current
 * thread. If there is no attached message loop, the return value will be 0.
 */
static PP_Resource GetCurrent(void)
{
    int i;
    PP_Resource r = 0;
    pthread_t curr = pthread_self();

    pthread_mutex_lock(&registry.lock);

    for(i = 0; i < registry.count && !r; i++)
        if(curr == registry.list[i].thread)
            r = registry.list[i].msg_loop;

    pthread_mutex_unlock(&registry.lock);

    return r;
};


/**
 * Sets the given message loop resource as being the associated message loop
 * for the currently running thread.
 *
 * You must call this function exactly once on a thread before making any
 * PPAPI calls. A message loop can only be attached to one thread, and the
 * message loop can not be changed later. The message loop will be attached
 * as long as the thread is running or until you quit with should_destroy
 * set to PP_TRUE.
 *
 * If this function fails, attempting to run the message loop will fail.
 * Note that you can still post work to the message loop: it will get queued
 * up should the message loop eventually be successfully attached and run.
 *
 * @return
 * - PP_OK: The message loop was successfully attached to the thread and is
 * ready to use.
 * - PP_ERROR_BADRESOURCE: The given message loop resource is invalid.
 * - PP_ERROR_INPROGRESS: The current thread already has a message loop
 * attached. This will always be the case for the main thread, which has
 * an implicit system-created message loop attached.
 * - PP_ERROR_WRONG_THREAD: The current thread type can not have a message
 * loop attached to it. See the interface level discussion about these
 * special threads, which include realtime audio threads.
 */
static int32_t AttachToCurrentThread(PP_Resource res)
{
    int32_t r = PP_OK;
    message_loop_t* msg_loop = (message_loop_t*)res_private(res);
    pthread_t curr = pthread_self();

    if(!msg_loop->thread)
    {
        msg_loop->thread = curr;

        pthread_mutex_lock(&registry.lock);

        if(registry.list)
            registry.list = (message_loop_map_t*)realloc(registry.list,
                (registry.count + 1) * sizeof(message_loop_map_t));
        else
            registry.list = (message_loop_map_t*)malloc(
                (registry.count + 1) * sizeof(message_loop_map_t));

        registry.list[registry.count].msg_loop = res;
        registry.list[registry.count].thread = curr;

        registry.count++;

        pthread_mutex_unlock(&registry.lock);
    }
    else
        r = PP_ERROR_INPROGRESS;

    LOG("r=%d", r);

    return r;
};


/**
 * Runs the thread message loop. Running the message loop is required for you
 * to get issued completion callbacks on the thread.
 *
 * The message loop identified by the argument must have been previously
 * successfully attached to the current thread.
 *
 * You may not run nested message loops. Since the main thread has an
 * implicit message loop that the system runs, you may not call Run on the
 * main thread.
 *
 * @return
 * - PP_OK: The message loop was successfully run. Note that on
 * success, the message loop will only exit when you call PostQuit().
 * - PP_ERROR_BADRESOURCE: The given message loop resource is invalid.
 * - PP_ERROR_WRONG_THREAD: You are attempting to run a message loop that
 * has not been successfully attached to the current thread. Call
 * AttachToCurrentThread().
 * - PP_ERROR_INPROGRESS: You are attempting to call Run in a nested
 * fashion (Run is already on the stack). This will occur if you attempt
 * to call run on the main thread's message loop (see above).
 */
static int32_t Run(PP_Resource res)
{
    message_loop_item_t* item;
    message_loop_t* msg_loop = (message_loop_t*)res_private(res);

    LOG("{%d}", res);

    while(!msg_loop->f_exit)
    {
        pthread_mutex_lock(&registry.lock);

        LOG1("{%d} msg_loop=%p, msg_loop->f_exit=%d, msg_loop->items=%p",
            res, msg_loop, msg_loop->f_exit, msg_loop->items);

        while(!msg_loop->f_exit && !msg_loop->items)
            pthread_cond_wait(&msg_loop->cond, &registry.lock);

        if(msg_loop->items)
        {
            item = msg_loop->items;
            msg_loop->items = item->next;
        };

        pthread_mutex_unlock(&registry.lock);

        if(item)
        {
            LOG1("%d", res);

            item->callback.func(item->callback.user_data, item->result);

            free(item);
            item = NULL;
        };
    };

    return 0;
};


/**
 * Schedules work to run on the given message loop. This may be called from
 * any thread. Posted work will be executed in the order it was posted when
 * the message loop is Run().
 *
 * @param message_loop The message loop resource.
 *
 * @param callback The completion callback to execute from the message loop.
 *
 * @param delay_ms The number of milliseconds to delay execution of the given
 * completion callback. Passing 0 means it will get queued normally and
 * executed in order.
 *
 *
 * The completion callback will be called with PP_OK as the "result" parameter
 * if it is run normally. It is good practice to check for PP_OK and return
 * early otherwise.
 *
 * The "required" flag on the completion callback is ignored. If there is an
 * error posting your callback, the error will be returned from PostWork and
 * the callback will never be run (because there is no appropriate place to
 * run your callback with an error without causing unexpected threading
 * problems). If you associate memory with the completion callback (for
 * example, you're using the C++ CompletionCallbackFactory), you will need to
 * free this or manually run the callback. See "Destruction and error
 * handling" above.
 *
 *
 * You can call this function before the message loop has started and the
 * work will get queued until the message loop is run. You can also post
 * work after the message loop has exited as long as should_destroy was
 * PP_FALSE. It will be queued until the next invocation of Run().
 *
 * @return
 * - PP_OK: The work was posted to the message loop's queue. As described
 * above, this does not mean that the work has been or will be executed
 * (if you never run the message loop after posting).
 * - PP_ERROR_BADRESOURCE: The given message loop resource is invalid.
 * - PP_ERROR_BADARGUMENT: The function pointer for the completion callback
 * is null (this will be the case if you pass PP_BlockUntilComplete()).
 * - PP_ERROR_FAILED: The message loop has been destroyed.
 */
static int32_t PostWork(PP_Resource message_loop, struct PP_CompletionCallback callback, int64_t delay_ms)
{
    LOG_TD;
    return PPB_MessageLoop_push(message_loop, callback, delay_ms, PP_OK);
};

int PPB_MessageLoop_push(PP_Resource message_loop, struct PP_CompletionCallback callback, int64_t delay_ms, int32_t result)
{
    int r, i;

    /* find main thread */
    pthread_mutex_lock(&registry.lock);

    LOG1("message_loop=%d", message_loop);

    /* find message loop for main thread */
    if(!message_loop)
    {

        for(r = -1, i = 0; i < registry.count && -1 == r; i++)
            if(PPB_MessageLoop_main_thread == registry.list[i].thread)
            {
                message_loop = registry.list[i].msg_loop;
                r = i;
            };
    };

    LOG1("message_loop=%d", message_loop);

    /* push into queue */
    if(message_loop <= 0)
    {
        r = PP_ERROR_BADRESOURCE;
        LOG("message loop not found");
    }
    else
    {
        message_loop_t* ctx = (message_loop_t*)res_private(message_loop);
        message_loop_item_t* item = (message_loop_item_t*)calloc(1, sizeof(message_loop_item_t));

        /* create item */
        item->callback = callback;
        item->delay_ms = delay_ms;
        item->result = result;
        item->now = _now_us() / 1000LL;
        item->next = ctx->items;

        /* append it */
        ctx->items = item;

        LOG1("{%d} msg_loop=%p, msg_loop->f_exit=%d, msg_loop->items=%p",
            message_loop, ctx, ctx->f_exit, ctx->items);

        pthread_cond_signal(&ctx->cond);

        r = 0;
    };

    pthread_mutex_unlock(&registry.lock);

    return r;
};


/**
 * Posts a quit message to the given message loop's work queue. Work posted
 * before that point will be processed before quitting.
 *
 * This may be called on the message loop registered for the current thread,
 * or it may be called on the message loop registered for another thread. It
 * is an error to attempt to PostQuit() the main thread loop.
 *
 * @param should_destroy Marks the message loop as being in a destroyed state
 * and prevents further posting of messages.
 *
 * If you quit a message loop without setting should_destroy, it will still
 * be attached to the thread and you can still run it again by calling Run()
 * again. If you destroy it, it will be detached from the current thread.
 *
 * @return
 * - PP_OK: The request to quit was successfully posted.
 * - PP_ERROR_BADRESOURCE: The message loop was invalid.
 * - PP_ERROR_WRONG_THREAD: You are attempting to quit the main thread.
 * The main thread's message loop is managed by the system and can't be
 * quit.
 */
static int32_t PostQuit(PP_Resource message_loop, PP_Bool should_destroy)
{
    LOG_NP;
    return 0;
};


/**
 * A message loop allows PPAPI calls to be issued on a thread. You may not
 * issue any API calls on a thread without creating a message loop. It also
 * allows you to post work to the message loop for a thread.
 *
 * To process work posted to the message loop, as well as completion callbacks
 * for asynchronous operations, you must run the message loop via Run().
 *
 * Note the system manages the lifetime of the instance (and all associated
 * resources). If the instance is deleted from the page, background threads may
 * suddenly see their PP_Resource handles become invalid. In this case, calls
 * will fail with PP_ERROR_BADRESOURCE. If you need to access data associated
 * with your instance, you will probably want to create some kind of threadsafe
 * proxy object that can handle asynchronous destruction of the instance object.
 *
 * Typical usage:
 *   On the main thread:
 *    - Create the thread yourself (using pthreads).
 *    - Create the message loop resource.
 *    - Pass the message loop resource to your thread's main function.
 *    - Call PostWork() on the message loop to run functions on the thread.
 *
 *   From the background thread's main function:
 *    - Call AttachToCurrentThread() with the message loop resource.
 *    - Call Run() with the message loop resource.
 *
 *   Your callbacks should look like this:
 *   @code
 *   void DoMyWork(void* user_data, int32_t status) {
 *     if (status != PP_OK) {
 *       Cleanup();  // e.g. free user_data.
 *       return;
 *     }
 *     ... do your work...
 *   }
 *   @endcode
 * For a C++ example, see ppapi/utility/threading/simple_thread.h
 *
 * (You can also create the message loop resource on the background thread,
 * but then the main thread will have no reference to it should you want to
 * call PostWork()).
 *
 *
 * THREAD HANDLING
 *
 * The main thread has an implicitly created message loop. The main thread is
 * the thread where PPP_InitializeModule and PPP_Instance functions are called.
 * You can retrieve a reference to this message loop by calling
 * GetForMainThread() or, if your code is on the main thread, GetCurrent() will
 * also work.
 *
 * Some special threads created by the system can not have message loops. In
 * particular, the background thread created for audio processing has this
 * requirement because it's intended to be highly responsive to keep up with
 * the realtime requirements of audio processing. You can not make PPAPI calls
 * from these threads.
 *
 * Once you associate a message loop with a thread, you don't have to keep a
 * reference to it. The system will hold a reference to the message loop for as
 * long as the thread is running. The current message loop can be retrieved
 * using the GetCurrent() function.
 *
 * It is legal to create threads in your plugin without message loops, but
 * PPAPI calls will fail unless explicitly noted in the documentation.
 *
 * You can create a message loop object on a thread and never actually run the
 * message loop. This will allow you to call blocking PPAPI calls (via
 * PP_BlockUntilComplete()). If you make any asynchronous calls, the callbacks
 * from those calls will be queued in the message loop and never run. The same
 * thing will happen if work is scheduled after the message loop exits and
 * the message loop is not run again.
 *
 *
 * DESTRUCTION AND ERROR HANDLING
 *
 * Often, your application will associate memory with completion callbacks. For
 * example, the C++ CompletionCallbackFactory has a small amount of
 * heap-allocated memory for each callback. This memory will be leaked if the
 * callback is never run. To avoid this memory leak, you need to be careful
 * about error handling and shutdown.
 *
 * There are a number of cases where posted callbacks will never be run:
 *
 *  - You tear down the thread (via pthreads) without "destroying" the message
 *    loop (via PostQuit with should_destroy = PP_TRUE). In this case, any
 *    tasks in the message queue will be lost.
 *
 *  - You create a message loop, post callbacks to it, and never run it.
 *
 *  - You quit the message loop via PostQuit with should_destroy set to
 *    PP_FALSE. In this case, the system will assume the message loop will be
 *    run again later and keep your tasks.
 *
 * To do proper shutdown, call PostQuit with should_destroy = PP_TRUE. This
 * will prohibit future work from being posted, and will allow the message loop
 * to run until all pending tasks are run.
 *
 * If you post a callback to a message loop that's been destroyed, or to an
 * invalid message loop, PostWork will return an error and will not run the
 * callback. This is true even for callbacks with the "required" flag set,
 * since the system may not even know what thread to issue the error callback
 * on.
 *
 * Therefore, you should check for errors from PostWork and destroy any
 * associated memory to avoid leaks. If you're using the C++
 * CompletionCallbackFactory, use the following pattern:
 * @code
 * pp::CompletionCallback callback = factory_.NewOptionalCallback(...);
 * int32_t result = message_loop.PostWork(callback);
 * if (result != PP_OK)
 *   callback.Run(result);
 * @endcode
 * This will run the callback with an error value, and assumes that the
 * implementation of your callback checks the "result" argument and returns
 * immediately on error.
 */
struct PPB_MessageLoop_1_0 PPB_MessageLoop_1_0_instance =
{
    .Create = Create,
    .GetForMainThread = GetForMainThread,
    .GetCurrent = GetCurrent,
    .AttachToCurrentThread = AttachToCurrentThread,
    .Run = Run,
    .PostWork = PostWork,
    .PostQuit = PostQuit,
};
