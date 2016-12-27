#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <dirent.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_errors.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_file_info.h>
#include <ppapi/c/ppb_file_io.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/dev/deprecated_bool.h>
#include <ppapi/c/private/ppb_flash_file.h>

#include "log.h"
#include "res.h"
#include "instance.h"

static int mkdir_p(char* path, int mode)
{
    int r;
    char* p;

    /* check empty string */
    if(0 == *path)
        return 0;

    for(r = 0, p = path + 1; (0 != (*p)) ; p++)
        if('/' == (*p))
        {
            *p = 0;

            r = mkdir(path, mode);

            *p = '/';
        };

    r = mkdir(path,mode);

    return ((0 == r) || (EEXIST == errno))?0:-errno;
};

static int32_t OpenFile_(PP_Resource file_ref_id, int32_t mode, PP_FileHandle* file)
{
    LOG_NP;
    return 0;
};

static int32_t QueryFile_(PP_Resource file_ref_id, struct PP_FileInfo* info)
{
    LOG_NP;
    return 0;
};

struct PPB_Flash_File_FileRef PPB_Flash_File_FileRef_instance =
{
    .OpenFile = OpenFile_,
    .QueryFile = QueryFile_,
};

// --------------------------------------------------------------------

static bool CreateThreadAdapterForInstance(PP_Instance instance)
{
    LOG_NP;
    return 0;
};

static void ClearThreadAdapterForInstance(PP_Instance instance)
{
    LOG_NP;
};

// Opens a file, returning a file descriptor (posix) or a HANDLE (win32) into
// file. The return value is the ppapi error, PP_OK if success, one of the
// PP_ERROR_* in case of failure.
static int32_t OpenFile(PP_Instance instance, const char* path, int32_t mode, PP_FileHandle* file)
{
    int r, m = 0;
    char dst[PATH_MAX];
    instance_t* pi = (instance_t*)res_private(instance);

    if((mode & PP_FILEOPENFLAG_READ) && !(mode & PP_FILEOPENFLAG_WRITE))
        m = O_RDONLY;
    else if (!(mode & PP_FILEOPENFLAG_READ) && (mode & PP_FILEOPENFLAG_WRITE))
        m = O_WRONLY;
    if ((mode & PP_FILEOPENFLAG_READ) && (mode & PP_FILEOPENFLAG_WRITE))
        m = O_RDWR;
    if (mode & PP_FILEOPENFLAG_CREATE)
        m |= O_CREAT;
    if (mode & PP_FILEOPENFLAG_TRUNCATE)
        m |= O_TRUNC;
    if (mode & PP_FILEOPENFLAG_EXCLUSIVE)
        m |= O_EXCL;
    if (mode & PP_FILEOPENFLAG_APPEND)
        m |= O_APPEND;

    snprintf(dst, sizeof(dst), "%s/%s", pi->paths.Local, path);

    r = open(dst, m | O_SYNC, 0666);

    LOG("r=%d", r);
    LOG("path=[%s]", path);
    LOG("dst=[%s]", dst);

    if(-1 == r)
    {
        r = errno;

        *file = PP_kInvalidFileHandle;

        if(ENOENT == r)
            return PP_ERROR_FILENOTFOUND;
        if(EACCES == r)
            return PP_ERROR_NOACCESS;
        return PP_ERROR_FAILED;
    };

    *file = r;

    return PP_OK;
};

static int32_t RenameFile(PP_Instance instance, const char* path_from, const char* path_to)
{
    int r;
    char dst[PATH_MAX], src[PATH_MAX];
    instance_t* pi = (instance_t*)res_private(instance);

    snprintf(src, sizeof(src), "%s/%s", pi->paths.Local, path_from);
    snprintf(dst, sizeof(dst), "%s/%s", pi->paths.Local, path_to);

    r = rename(src, dst);

    LOG("r=%d", r);
    LOG("path_from=[%s]", path_from);
    LOG("path_to=[%s]", path_to);

    if(r)
    {
        r = errno;

        if(ENOENT == r)
            r = PP_ERROR_FILENOTFOUND;
        else if(EACCES == r)
            r = PP_ERROR_NOACCESS;
        else
            r = PP_ERROR_FAILED;
    };

    return r;
};

static int32_t DeleteFileOrDir(PP_Instance instance, const char* path, PP_Bool recursive)
{
    char dst[PATH_MAX];
    instance_t* pi = (instance_t*)res_private(instance);

    snprintf(dst, sizeof(dst), "%s/%s", pi->paths.Local, path);

    LOG("path=[%s], recursive=%d", path, recursive);

    return 0;
};

static int32_t CreateDir(PP_Instance instance, const char* path)
{
    char dst[PATH_MAX];
    instance_t* pi = (instance_t*)res_private(instance);

    snprintf(dst, sizeof(dst), "%s/%s", pi->paths.Local, path);

    LOG("path=[%s]", path);
    mkdir_p(dst, 0666);

    return 0;
};

// Queries information about a file. The return value is the ppapi error,
// PP_OK if success, one of the PP_ERROR_* in case of failure.
static int32_t QueryFile(PP_Instance instance, const char* path, struct PP_FileInfo* info)
{
    int r;
    struct stat st;
    char dst[PATH_MAX];
    instance_t* pi = (instance_t*)res_private(instance);

    snprintf(dst, sizeof(dst), "%s/%s", pi->paths.Local, path);

    r = stat(dst, &st);

    LOG("r=%d", r);
    LOG("path=[%s]", path);
    LOG("dst=[%s]", dst);

    if(-1 == r)
    {
        r = errno;

        if(ENOENT == r)
            return PP_ERROR_FILENOTFOUND;
        if(EACCES == r)
            return PP_ERROR_NOACCESS;
        return PP_ERROR_FAILED;
    };

    info->size = st.st_size;
    if(S_ISDIR(st.st_mode))
        info->type = PP_FILETYPE_DIRECTORY;
    else if(S_ISREG(st.st_mode))
        info->type = PP_FILETYPE_REGULAR;
    else if(S_ISLNK(st.st_mode))
    {
        info->type = PP_FILETYPE_REGULAR;
    }
    else
        info->type = PP_FILETYPE_OTHER;
    info->creation_time = st.st_ctime;
    info->last_access_time = st.st_atime;
    info->last_modified_time = st.st_mtime;
    info->system_type = PP_FILESYSTEMTYPE_LOCALPERSISTENT;

    return PP_OK;
};

// Gets the list of files contained in a directory. The return value is the
// ppapi error, PP_OK if success, one of the PP_ERROR_* in case of failure. If
// non-NULL, the returned contents should be freed with FreeDirContents.
static int32_t GetDirContents(PP_Instance instance, const char* path, struct PP_DirContents_Dev** contents)
{
    int r;
    DIR *dir;
    struct dirent *ent;
    char dst[PATH_MAX];
    struct PP_DirContents_Dev* result = NULL;
    instance_t* pi = (instance_t*)res_private(instance);

    snprintf(dst, sizeof(dst), "%s/%s", pi->paths.Local, path);

    dir = opendir(dst);
    if(!dir)
    {
        r = errno;

        if(ENOENT == r)
            return PP_ERROR_FILENOTFOUND;
        if(EACCES == r)
            return PP_ERROR_NOACCESS;
        return PP_ERROR_FAILED;
    };

    while((ent = readdir(dir)))
    {
        struct stat st;
        struct PP_DirEntry_Dev e;

        /* build new name */
        snprintf(dst, sizeof(dst), "%s/%s/%s", pi->paths.Local, path, ent->d_name);

        /* test it */
        if(stat(dst, &st))
        {
            LOG("stat([%s]) failed, errno=%d", dst, errno);
            continue;
        };

        /* fill PP_DirEntry_Dev */
        e.is_dir = S_ISDIR(st.st_mode);
        e.name = strdup(ent->d_name);

        /* insert */
        if(!result)
            result = (struct PP_DirContents_Dev*)calloc(1, sizeof(struct PP_DirContents_Dev));
        if(!result->entries)
            result->entries = (struct PP_DirEntry_Dev*)malloc(sizeof(struct PP_DirEntry_Dev));
        else
            result->entries = (struct PP_DirEntry_Dev*)realloc(result->entries, (result->count + 1) * sizeof(struct PP_DirEntry_Dev));

        result->entries[result->count] = e;
        result->count++;
    };

    *contents = result;

    closedir(dir);

    return 0;
};

// Frees the data allocated by GetDirContents.
static void FreeDirContents(PP_Instance instance, struct PP_DirContents_Dev* contents)
{
    if(contents && contents->entries)
        free(contents->entries);

    if(contents)
        free(contents);
};

static int32_t CreateTemporaryFile(PP_Instance instance, PP_FileHandle* file)
{
    LOG_NP;
    return 0;
};

struct PPB_Flash_File_ModuleLocal_3_0 PPB_Flash_File_ModuleLocal_3_0_instance =
{
    .CreateThreadAdapterForInstance = CreateThreadAdapterForInstance,
    .ClearThreadAdapterForInstance = ClearThreadAdapterForInstance,
    .OpenFile = OpenFile,
    .RenameFile = RenameFile,
    .DeleteFileOrDir = DeleteFileOrDir,
    .CreateDir = CreateDir,
    .QueryFile = QueryFile,
    .GetDirContents = GetDirContents,
    .FreeDirContents = FreeDirContents,
    .CreateTemporaryFile = CreateTemporaryFile,
};
