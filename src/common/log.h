#ifndef LOG_H
#define LOG_H

#define LOG_MESSAGE(LEVEL, FMT, ...) \
    log_message(LEVEL,  __FILE__, __LINE__, __FUNCTION__, FMT, ##__VA_ARGS__)

#define LOG_E(FMT, ...) LOG_MESSAGE( 0, FMT, ##__VA_ARGS__)
#define LOG_N(FMT, ...) LOG_MESSAGE(10, FMT, ##__VA_ARGS__)
#define LOG_D(FMT, ...) LOG_MESSAGE(20, FMT, ##__VA_ARGS__)
#define LOG_T(FMT, ...) LOG_MESSAGE(30, FMT, ##__VA_ARGS__)

#define LOG_NP          LOG_E("%s", "NOT IMPLEMENTED")
#define LOG_TD          LOG_E("%s", "TODO")
#define LOG_PL          LOG_E("%s", "");

void log_level(int level);

void log_message(int level, const char* file, const int line, const char* function, const char *fmt, ...)
#ifdef __GNUC__
__attribute__ ((__format__ (__printf__, 5, 6)))
#endif
;

#endif /* LOG_H */
