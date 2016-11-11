#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_file_ref.h>

#include "log.h"

static PP_Resource Create(PP_Resource file_system, const char* path)
{
    LOG_NP;
    return 0;
};

static PP_Bool IsFileRef(PP_Resource resource)
{
    LOG_NP;
    return 0;
};

static PP_FileSystemType GetFileSystemType(PP_Resource file_ref)
{
    LOG_NP;
    return 0;
};

static struct PP_Var GetName(PP_Resource file_ref)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

static struct PP_Var GetPath(PP_Resource file_ref)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

static PP_Resource GetParent(PP_Resource file_ref)
{
    LOG_NP;
    return 0;
};

static int32_t MakeDirectory(PP_Resource directory_ref,
    int32_t make_directory_flags, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

static int32_t Touch(PP_Resource file_ref, PP_Time last_access_time,
    PP_Time last_modified_time, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

static int32_t Delete(PP_Resource file_ref, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

static int32_t Rename(PP_Resource file_ref,
    PP_Resource new_file_ref, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

static int32_t Query(PP_Resource file_ref,
    struct PP_FileInfo* info, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

static int32_t ReadDirectoryEntries(PP_Resource file_ref,
    struct PP_ArrayOutput output, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

struct PPB_FileRef_1_2 PPB_FileRef_1_2_instance =
{
    .Create = Create,
    .IsFileRef = IsFileRef,
    .GetFileSystemType = GetFileSystemType,
    .GetName = GetName,
    .GetPath = GetPath,
    .GetParent = GetParent,
    .MakeDirectory = MakeDirectory,
    .Touch = Touch,
    .Delete = Delete,
    .Rename = Rename,
    .Query = Query,
    .ReadDirectoryEntries = ReadDirectoryEntries,
};
