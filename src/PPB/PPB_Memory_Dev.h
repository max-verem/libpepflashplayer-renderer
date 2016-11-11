#ifndef PPB_Memory_Dev_h
#define PPB_Memory_Dev_h

void* MemAlloc(uint32_t num_bytes);
void MemFree(void* ptr);

#endif /* PPB_Memory_Dev_h */