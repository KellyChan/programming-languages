#ifndef STRING_HELPER_H
#define STRING_HELPER_H

#include <string>
#include <stdio.h>
#include <stdlib.h>


#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)  // Windows

#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif

#ifndef STRCASECMP
#define STRCASECMP _stricmp
#endif

#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif

#ifndef STRCPY
#define STRCPY(sFilePath, nLength, sPath) strcpy_s(sFilePath, nLength, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) fopen_s(&fHandle, filename, mode)
#endif

#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result != 0)
#endif

#ifndef SSCANF
#define SSCANF sscanf_s
#endif

#else  // Linux

#include <string.h>
#include <strings.h>

#ifndef STRCASECMP
#define STRCASECMP strcasecmp
#endif

#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif

#ifndef STRCPY
#define STRCPY(sFilePath, nLength, sPath) strcpy(sFilePath, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) (fHandle = fopen(filename, mode))
#endif

#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result == NULL)
#endif

#ifndef SSCANF
#define SSCANF sscanf
#endif

#endif


inline int stringRemoveDelimiter(char delimiter, const char *string)
{
    int string_start = 0;
    
    while (string[string_start] == delimiter)
    {
        string_start++;
    }

    if (string_start >= (int)strlen(string)-1)
    {
        return 0;
    }

    return string_start;
}


inline int checkCmdLineFlag(const int argc, const char **argv, const char *string_ref)
{
    bool bFound = false;

    if (argc >= 1)
    {
        for (int i = 1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
    
            const char *equal_pos = strchr(string_argv, '=');
            int argv_length = (int)(equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

            int length = (int)strlen(string_ref);

            if (length == argv_length && !STRNCASECMP(string_argv, string_ref, length))
            {
                bFound = true;
                continue;
            }
        }
    }

    return (int)bFound;
}


inline int getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref)
{
    bool bFound = false;
    int value = -1;

    if (argc >= 1)
    {
        for (int i = 1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length = (int)strlen(string_ref);

            if (!STRNCASECMP(string_argv, string_ref, length))
            {
                if (length+1 <= (int)strlen(string_argv))
                {
                    int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                    value = atoi(&string_argv[length + auto_inc]);
                }
                else
                {
                    value = 0;
                }

                bFound = true;
                continue;
            }
        }
    }

    if (bFound)
    {
        return value;
    }
    else
    {
        return 0;
    }
}

#endif
