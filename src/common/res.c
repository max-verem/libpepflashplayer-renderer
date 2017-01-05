#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

#include "res.h"
#include "log.h"

#define RELEASED_SIZE_GROW 1024
#define ACTIVE_SIZE_GROW 1024

typedef struct res_desc
{
    int ref;
    void* priv;
    size_t size;
    void* interface;
    res_destructor_t destructor;
} res_t;

static pthread_mutex_t lock;

static struct
{
    PP_Resource* list;
    int size, count;
} released;

static struct
{
    res_t** list;
    int size, count;
} active;

static int zero_fake_resource = -1;

int res_begin()
{
    pthread_mutexattr_t mta;

    pthread_mutexattr_init( &mta );
    pthread_mutexattr_settype( &mta, PTHREAD_MUTEX_RECURSIVE );
    pthread_mutex_init( &lock, &mta );
    pthread_mutexattr_destroy( &mta );

    released.size = released.count = 0;
    released.list = malloc(0);

    active.size = active.count = 0;
    active.list = malloc(0);

    zero_fake_resource = res_create(0, NULL, NULL);

    return 0;
};

static int res_destroy(PP_Resource res);

int res_end()
{
    int i, c;

    pthread_mutex_lock(&lock);

    res_release(zero_fake_resource);

    LOG_N("------------------------ leaked resources ------------------");
    for(c = 0, i = 0; i < active.size; i++)
        if(active.list[i])
        {
            LOG_N("res=%d, ref=%d, size=%d, priv=%p", i,
                active.list[i]->ref, (int)active.list[i]->size, active.list[i]->priv);
            c++;
        };
    LOG_N("leaked: %d", c);
    LOG_N("------------------------ /leaked resources -----------------");


    for(i = 0; i < active.size; i++)
        if(active.list[i])
            res_destroy((PP_Resource)i);

    free(active.list);

    free(released.list);

    pthread_mutex_unlock(&lock);

    pthread_mutex_destroy(&lock);

    return 0;
};

PP_Resource res_create(size_t private_size, void* interface, res_destructor_t destructor)
{
    PP_Resource res = 0;
    res_t* r = calloc(1, sizeof(res_t));

    LOG_T("private_size=%zd", private_size);

    r->ref = 1;
    r->priv = calloc(1, private_size);
    r->size = private_size;
    r->interface = interface;
    r->destructor = destructor;

    pthread_mutex_lock(&lock);

    if(released.count)
    {
        released.count--;
        res = released.list[released.count];
        active.list[res] = r;
        active.count++;
    }
    else
    {
        if(active.count == active.size)
        {
            active.size += ACTIVE_SIZE_GROW;
            active.list = realloc(active.list, active.size * sizeof(res_t*));
        };

        res = active.count;
        active.list[res] = r;
        active.count++;
    };

    pthread_mutex_unlock(&lock);

    LOG_T("private_size=%zd, res=%d", private_size, res);

    return res;
};

int res_add_ref(PP_Resource res)
{
    int r = -ENOENT;

    LOG_T("{%d}", res);

    pthread_mutex_lock(&lock);

    if(res < active.size && active.list[res])
        r = ++active.list[res]->ref;

    pthread_mutex_unlock(&lock);

    return r;
};

int res_release(PP_Resource res)
{
    int r = -ENOENT;

    LOG_T("{%d}", res);

    pthread_mutex_lock(&lock);

    if(res < active.size && active.list[res])
    {
        r = --active.list[res]->ref;
        if(!r)
            res_destroy(res);
    };

    pthread_mutex_unlock(&lock);

    return r;
}

static int res_destroy(PP_Resource res)
{
    res_t* r;

    /* reallocate release list */
    if(released.count == released.size)
    {
        released.size += RELEASED_SIZE_GROW;
        released.list = realloc(released.list, released.size * sizeof(PP_Resource));
    };

    /* save released */
    released.list[released.count++] = res;

    /* fetch from list */
    r = active.list[res];
    active.list[res] = NULL;
    active.count--;

    /* free resource */
    if(r->destructor)
        r->destructor(r->priv);
    free(r->priv);
    free(r);

    return 0;
};

void* res_private(PP_Resource res)
{
    void* priv = NULL;

    pthread_mutex_lock(&lock);

    if(res < active.size && active.list[res])
        priv = active.list[res]->priv;

    pthread_mutex_unlock(&lock);

    return priv;
};

int res_private_size(PP_Resource res)
{
    int r = 0;

    pthread_mutex_lock(&lock);

    if(res < active.size && active.list[res])
        r = active.list[res]->size;
    else
        r = -ENOENT;

    pthread_mutex_unlock(&lock);

    return r;
};

void* res_interface(PP_Resource res)
{
    void* interface = NULL;

    pthread_mutex_lock(&lock);

    if(res < active.size && active.list[res])
        interface = active.list[res]->interface;

    pthread_mutex_unlock(&lock);

    return interface;
};
