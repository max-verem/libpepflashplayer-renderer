#ifndef IF_H
#define IF_H

typedef struct if_desc
{
    const char* name;
    const void* ptr;
    const void* destructor;
} if_t;

const if_t* if_find(const char* name);

const void* get_browser_interface_proc(const char* interface_name);

#endif /* IF_H */
