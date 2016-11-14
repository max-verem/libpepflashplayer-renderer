#ifndef LOG_H
#define LOG_H

#define LOG1(FMT, ...)
#define LOG(FMT, ...) fprintf(stderr, "%s:%d:%s " FMT "\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__);
#define LOG_D LOG
#define LOG_NP LOG("NOT IMPLEMENTED")
#define LOG_TD LOG("TODO")

#endif /* LOG_H */
