#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_opengles2.h>

#include "log.h"

static void ActiveTexture(PP_Resource context, GLenum texture)
{
    LOG_NP;
};

static void AttachShader(PP_Resource context, GLuint program, GLuint shader)
{
    LOG_NP;
};

static void BindAttribLocation(PP_Resource context,
                             GLuint program,
                             GLuint index,
                             const char* name)
{
    LOG_NP;
};

static void BindBuffer(PP_Resource context, GLenum target, GLuint buffer)
{
    LOG_NP;
};

static void BindFramebuffer(PP_Resource context,
                          GLenum target,
                          GLuint framebuffer)
{
    LOG_NP;
};

static void BindRenderbuffer(PP_Resource context,
                           GLenum target,
                           GLuint renderbuffer)
{
    LOG_NP;
};

static void BindTexture(PP_Resource context, GLenum target, GLuint texture)
{
    LOG_NP;
};

static void BlendColor(PP_Resource context,
                     GLclampf red,
                     GLclampf green,
                     GLclampf blue,
                     GLclampf alpha)
{
    LOG_NP;
};

static void BlendEquation(PP_Resource context, GLenum mode)
{
    LOG_NP;
};

static void BlendEquationSeparate(PP_Resource context,
                                GLenum modeRGB,
                                GLenum modeAlpha)
{
    LOG_NP;
};

static void BlendFunc(PP_Resource context, GLenum sfactor, GLenum dfactor)
{
    LOG_NP;
};

static void BlendFuncSeparate(PP_Resource context,
                            GLenum srcRGB,
                            GLenum dstRGB,
                            GLenum srcAlpha,
                            GLenum dstAlpha)
{
    LOG_NP;
};

static void BufferData(PP_Resource context,
                     GLenum target,
                     GLsizeiptr size,
                     const void* data,
                     GLenum usage)
{
    LOG_NP;
};

static void BufferSubData(PP_Resource context,
                        GLenum target,
                        GLintptr offset,
                        GLsizeiptr size,
                        const void* data)
{
    LOG_NP;
};

static GLenum CheckFramebufferStatus(PP_Resource context, GLenum target)
{
    LOG_NP;
};

static void Clear(PP_Resource context, GLbitfield mask)
{
    LOG_NP;
};

static void ClearColor(PP_Resource context,
                     GLclampf red,
                     GLclampf green,
                     GLclampf blue,
                     GLclampf alpha)
{
    LOG_NP;
};

static void ClearDepthf(PP_Resource context, GLclampf depth)
{
    LOG_NP;
};

static void ClearStencil(PP_Resource context, GLint s)
{
    LOG_NP;
};

static void ColorMask(PP_Resource context,
                    GLboolean red,
                    GLboolean green,
                    GLboolean blue,
                    GLboolean alpha)
{
    LOG_NP;
};

static void CompileShader(PP_Resource context, GLuint shader)
{
    LOG_NP;
};

static void CompressedTexImage2D(PP_Resource context,
                               GLenum target,
                               GLint level,
                               GLenum internalformat,
                               GLsizei width,
                               GLsizei height,
                               GLint border,
                               GLsizei imageSize,
                               const void* data)
{
    LOG_NP;
};

static void CompressedTexSubImage2D(PP_Resource context,
                                  GLenum target,
                                  GLint level,
                                  GLint xoffset,
                                  GLint yoffset,
                                  GLsizei width,
                                  GLsizei height,
                                  GLenum format,
                                  GLsizei imageSize,
                                  const void* data)
{
    LOG_NP;
};

static void CopyTexImage2D(PP_Resource context,
                         GLenum target,
                         GLint level,
                         GLenum internalformat,
                         GLint x,
                         GLint y,
                         GLsizei width,
                         GLsizei height,
                         GLint border)
{
    LOG_NP;
};

static void CopyTexSubImage2D(PP_Resource context,
                            GLenum target,
                            GLint level,
                            GLint xoffset,
                            GLint yoffset,
                            GLint x,
                            GLint y,
                            GLsizei width,
                            GLsizei height)
{
    LOG_NP;
};

static GLuint CreateProgram(PP_Resource context)
{
    LOG_NP;
    return 0;
};

static GLuint CreateShader(PP_Resource context, GLenum type)
{
    LOG_NP;
};

static void CullFace(PP_Resource context, GLenum mode)
{
    LOG_NP;
};

static void DeleteBuffers(PP_Resource context, GLsizei n, const GLuint* buffers)
{
    LOG_NP;
};

static void DeleteFramebuffers(PP_Resource context,
                             GLsizei n,
                             const GLuint* framebuffers)
{
    LOG_NP;
};

static void DeleteProgram(PP_Resource context, GLuint program)
{
    LOG_NP;
};

static void DeleteRenderbuffers(PP_Resource context,
                              GLsizei n,
                              const GLuint* renderbuffers)
{
    LOG_NP;
};

static void DeleteShader(PP_Resource context, GLuint shader)
{
    LOG_NP;
};

static void DeleteTextures(PP_Resource context,
                         GLsizei n,
                         const GLuint* textures)
{
    LOG_NP;
};

static void DepthFunc(PP_Resource context, GLenum func)
{
    LOG_NP;
};

static void DepthMask(PP_Resource context, GLboolean flag)
{
    LOG_NP;
};

static void DepthRangef(PP_Resource context, GLclampf zNear, GLclampf zFar)
{
    LOG_NP;
};

static void DetachShader(PP_Resource context, GLuint program, GLuint shader)
{
    LOG_NP;
};

static void Disable(PP_Resource context, GLenum cap)
{
    LOG_NP;
};

static void DisableVertexAttribArray(PP_Resource context, GLuint index)
{
    LOG_NP;
};

static void DrawArrays(PP_Resource context,
                     GLenum mode,
                     GLint first,
                     GLsizei count)
{
    LOG_NP;
};

static void DrawElements(PP_Resource context,
                       GLenum mode,
                       GLsizei count,
                       GLenum type,
                       const void* indices)
{
    LOG_NP;
};

static void Enable(PP_Resource context, GLenum cap)
{
    LOG_NP;
};

static void EnableVertexAttribArray(PP_Resource context, GLuint index)
{
    LOG_NP;
};

static void Finish(PP_Resource context)
{
    LOG_NP;
};

static void Flush(PP_Resource context)
{
    LOG_NP;
};

static void FramebufferRenderbuffer(PP_Resource context,
                                  GLenum target,
                                  GLenum attachment,
                                  GLenum renderbuffertarget,
                                  GLuint renderbuffer)
{
    LOG_NP;
};

static void FramebufferTexture2D(PP_Resource context,
                               GLenum target,
                               GLenum attachment,
                               GLenum textarget,
                               GLuint texture,
                               GLint level)
{
    LOG_NP;
};

static void FrontFace(PP_Resource context, GLenum mode)
{
    LOG_NP;
};

static void GenBuffers(PP_Resource context, GLsizei n, GLuint* buffers)
{
    LOG_NP;
};

static void GenerateMipmap(PP_Resource context, GLenum target)
{
    LOG_NP;
};

static void GenFramebuffers(PP_Resource context, GLsizei n, GLuint* framebuffers)
{
    LOG_NP;
};

static void GenRenderbuffers(PP_Resource context,
                           GLsizei n,
                           GLuint* renderbuffers)
{
    LOG_NP;
};

static void GenTextures(PP_Resource context, GLsizei n, GLuint* textures)
{
    LOG_NP;
};

static void GetActiveAttrib(PP_Resource context,
                          GLuint program,
                          GLuint index,
                          GLsizei bufsize,
                          GLsizei* length,
                          GLint* size,
                          GLenum* type,
                          char* name)
{
    LOG_NP;
};

static void GetActiveUniform(PP_Resource context,
                           GLuint program,
                           GLuint index,
                           GLsizei bufsize,
                           GLsizei* length,
                           GLint* size,
                           GLenum* type,
                           char* name)
{
    LOG_NP;
};

static void GetAttachedShaders(PP_Resource context,
                             GLuint program,
                             GLsizei maxcount,
                             GLsizei* count,
                             GLuint* shaders)
{
    LOG_NP;
};

static GLint GetAttribLocation(PP_Resource context,
                             GLuint program,
                             const char* name)
{
    LOG_NP;
    return 0;
};

static void GetBooleanv(PP_Resource context, GLenum pname, GLboolean* params)
{
    LOG_NP;
};

static void GetBufferParameteriv(PP_Resource context,
                               GLenum target,
                               GLenum pname,
                               GLint* params)
{
    LOG_NP;
};

static GLenum GetError(PP_Resource context)
{
    LOG_NP;
};

static void GetFloatv(PP_Resource context, GLenum pname, GLfloat* params)
{
    LOG_NP;
};

static void GetFramebufferAttachmentParameteriv(PP_Resource context,
                                              GLenum target,
                                              GLenum attachment,
                                              GLenum pname,
                                              GLint* params)
{
    LOG_NP;
};

static void GetIntegerv(PP_Resource context, GLenum pname, GLint* params)
{
    LOG_NP;
};

static void GetProgramiv(PP_Resource context,
                       GLuint program,
                       GLenum pname,
                       GLint* params)
{
    LOG_NP;
};

static void GetProgramInfoLog(PP_Resource context,
                            GLuint program,
                            GLsizei bufsize,
                            GLsizei* length,
                            char* infolog)
{
    LOG_NP;
};

static void GetRenderbufferParameteriv(PP_Resource context,
                                     GLenum target,
                                     GLenum pname,
                                     GLint* params)
{
    LOG_NP;
};

static void GetShaderiv(PP_Resource context,
                      GLuint shader,
                      GLenum pname,
                      GLint* params)
{
    LOG_NP;
};

static void GetShaderInfoLog(PP_Resource context,
                           GLuint shader,
                           GLsizei bufsize,
                           GLsizei* length,
                           char* infolog)
{
    LOG_NP;
};

static void GetShaderPrecisionFormat(PP_Resource context,
                                   GLenum shadertype,
                                   GLenum precisiontype,
                                   GLint* range,
                                   GLint* precision)
{
    LOG_NP;
};

static void GetShaderSource(PP_Resource context,
                          GLuint shader,
                          GLsizei bufsize,
                          GLsizei* length,
                          char* source)
{
    LOG_NP;
};

static const GLubyte* GetString(PP_Resource context, GLenum name)
{
    LOG_NP;
    return 0;
};

static void GetTexParameterfv(PP_Resource context,
                            GLenum target,
                            GLenum pname,
                            GLfloat* params)
{
    LOG_NP;
};

static void GetTexParameteriv(PP_Resource context,
                            GLenum target,
                            GLenum pname,
                            GLint* params)
{
    LOG_NP;
};

static void GetUniformfv(PP_Resource context,
                       GLuint program,
                       GLint location,
                       GLfloat* params)
{
    LOG_NP;
};

static void GetUniformiv(PP_Resource context,
                       GLuint program,
                       GLint location,
                       GLint* params)
{
    LOG_NP;
};

static GLint GetUniformLocation(PP_Resource context,
                              GLuint program,
                              const char* name)
{
    LOG_NP;
    return 0;
};

static void GetVertexAttribfv(PP_Resource context,
                            GLuint index,
                            GLenum pname,
                            GLfloat* params)
{
    LOG_NP;
};

static void GetVertexAttribiv(PP_Resource context,
                            GLuint index,
                            GLenum pname,
                            GLint* params)
{
    LOG_NP;
};

static void GetVertexAttribPointerv(PP_Resource context,
                                  GLuint index,
                                  GLenum pname,
                                  void** pointer)
{
    LOG_NP;
};

static void Hint(PP_Resource context, GLenum target, GLenum mode)
{
    LOG_NP;
};

static GLboolean IsBuffer(PP_Resource context, GLuint buffer)
{
    LOG_NP;
    return 0;
};

static GLboolean IsEnabled(PP_Resource context, GLenum cap)
{
    LOG_NP;
    return 0;
};

static GLboolean IsFramebuffer(PP_Resource context, GLuint framebuffer)
{
    LOG_NP;
    return 0;
};

static GLboolean IsProgram(PP_Resource context, GLuint program)
{
    LOG_NP;
    return 0;
};

static GLboolean IsRenderbuffer(PP_Resource context, GLuint renderbuffer)
{
    LOG_NP;
    return 0;
};

static GLboolean IsShader(PP_Resource context, GLuint shader)
{
    LOG_NP;
    return 0;
};

static GLboolean IsTexture(PP_Resource context, GLuint texture)
{
    LOG_NP;
    return 0;
};

static void LineWidth(PP_Resource context, GLfloat width)
{
    LOG_NP;
};

static void LinkProgram(PP_Resource context, GLuint program)
{
    LOG_NP;
};

static void PixelStorei(PP_Resource context, GLenum pname, GLint param)
{
    LOG_NP;
};

static void PolygonOffset(PP_Resource context, GLfloat factor, GLfloat units)
{
    LOG_NP;
};

static void ReadPixels(PP_Resource context,
                     GLint x,
                     GLint y,
                     GLsizei width,
                     GLsizei height,
                     GLenum format,
                     GLenum type,
                     void* pixels)
{
    LOG_NP;
};

static void ReleaseShaderCompiler(PP_Resource context)
{
    LOG_NP;
};

static void RenderbufferStorage(PP_Resource context,
                              GLenum target,
                              GLenum internalformat,
                              GLsizei width,
                              GLsizei height)
{
    LOG_NP;
};

static void SampleCoverage(PP_Resource context, GLclampf value, GLboolean invert)
{
    LOG_NP;
};

static void Scissor(PP_Resource context,
                  GLint x,
                  GLint y,
                  GLsizei width,
                  GLsizei height)
{
    LOG_NP;
};

static void ShaderBinary(PP_Resource context,
                       GLsizei n,
                       const GLuint* shaders,
                       GLenum binaryformat,
                       const void* binary,
                       GLsizei length)
{
    LOG_NP;
};

static void ShaderSource(PP_Resource context,
                       GLuint shader,
                       GLsizei count,
                       const char** str,
                       const GLint* length)
{
    LOG_NP;
};

static void StencilFunc(PP_Resource context, GLenum func, GLint ref, GLuint mask)
{
    LOG_NP;
};

static void StencilFuncSeparate(PP_Resource context,
                              GLenum face,
                              GLenum func,
                              GLint ref,
                              GLuint mask)
{
    LOG_NP;
};

static void StencilMask(PP_Resource context, GLuint mask)
{
    LOG_NP;
};

static void StencilMaskSeparate(PP_Resource context, GLenum face, GLuint mask)
{
    LOG_NP;
};

static void StencilOp(PP_Resource context,
                    GLenum fail,
                    GLenum zfail,
                    GLenum zpass)
{
    LOG_NP;
};

static void StencilOpSeparate(PP_Resource context,
                            GLenum face,
                            GLenum fail,
                            GLenum zfail,
                            GLenum zpass)
{
    LOG_NP;
};

static void TexImage2D(PP_Resource context,
                     GLenum target,
                     GLint level,
                     GLint internalformat,
                     GLsizei width,
                     GLsizei height,
                     GLint border,
                     GLenum format,
                     GLenum type,
                     const void* pixels)
{
    LOG_NP;
};

static void TexParameterf(PP_Resource context,
                        GLenum target,
                        GLenum pname,
                        GLfloat param)
{
    LOG_NP;
};

static void TexParameterfv(PP_Resource context,
                         GLenum target,
                         GLenum pname,
                         const GLfloat* params)
{
    LOG_NP;
};

static void TexParameteri(PP_Resource context,
                        GLenum target,
                        GLenum pname,
                        GLint param)
{
    LOG_NP;
};

static void TexParameteriv(PP_Resource context,
                         GLenum target,
                         GLenum pname,
                         const GLint* params)
{
    LOG_NP;
};

static void TexSubImage2D(PP_Resource context,
                        GLenum target,
                        GLint level,
                        GLint xoffset,
                        GLint yoffset,
                        GLsizei width,
                        GLsizei height,
                        GLenum format,
                        GLenum type,
                        const void* pixels)
{
    LOG_NP;
};

static void Uniform1f(PP_Resource context, GLint location, GLfloat x)
{
    LOG_NP;
};

static void Uniform1fv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLfloat* v)
{
    LOG_NP;
};

static void Uniform1i(PP_Resource context, GLint location, GLint x)
{
    LOG_NP;
};

static void Uniform1iv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLint* v)
{
    LOG_NP;
};

static void Uniform2f(PP_Resource context, GLint location, GLfloat x, GLfloat y)
{
    LOG_NP;
};

static  void Uniform2fv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLfloat* v)
{
    LOG_NP;
};

static void Uniform2i(PP_Resource context, GLint location, GLint x, GLint y)
{
    LOG_NP;
};

static void Uniform2iv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLint* v)
{
    LOG_NP;
};

static void Uniform3f(PP_Resource context,
                    GLint location,
                    GLfloat x,
                    GLfloat y,
                    GLfloat z)
{
    LOG_NP;
};

static void Uniform3fv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLfloat* v)
{
    LOG_NP;
};

static void Uniform3i(PP_Resource context,
                    GLint location,
                    GLint x,
                    GLint y,
                    GLint z)
{
    LOG_NP;
};

static void Uniform3iv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLint* v)
{
    LOG_NP;
};

static void Uniform4f(PP_Resource context,
                    GLint location,
                    GLfloat x,
                    GLfloat y,
                    GLfloat z,
                    GLfloat w)
{
    LOG_NP;
};

static void Uniform4fv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLfloat* v)
{
    LOG_NP;
};

static void Uniform4i(PP_Resource context,
                    GLint location,
                    GLint x,
                    GLint y,
                    GLint z,
                    GLint w)
{
    LOG_NP;
};

static void Uniform4iv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLint* v)
{
    LOG_NP;
};

static void UniformMatrix2fv(PP_Resource context,
                           GLint location,
                           GLsizei count,
                           GLboolean transpose,
                           const GLfloat* value)
{
    LOG_NP;
};

static void UniformMatrix3fv(PP_Resource context,
                           GLint location,
                           GLsizei count,
                           GLboolean transpose,
                           const GLfloat* value)
{
    LOG_NP;
};

static void UniformMatrix4fv(PP_Resource context,
                           GLint location,
                           GLsizei count,
                           GLboolean transpose,
                           const GLfloat* value)
{
    LOG_NP;
};

static void UseProgram(PP_Resource context, GLuint program)
{
    LOG_NP;
};

static void ValidateProgram(PP_Resource context, GLuint program)
{
    LOG_NP;
};

static void VertexAttrib1f(PP_Resource context, GLuint indx, GLfloat x)
{
    LOG_NP;
};

static void VertexAttrib1fv(PP_Resource context,
                          GLuint indx,
                          const GLfloat* values)
{
    LOG_NP;
};

static void VertexAttrib2f(PP_Resource context,
                         GLuint indx,
                         GLfloat x,
                         GLfloat y)
{
    LOG_NP;
};

static void VertexAttrib2fv(PP_Resource context,
                          GLuint indx,
                          const GLfloat* values)
{
    LOG_NP;
};

static void VertexAttrib3f(PP_Resource context,
                         GLuint indx,
                         GLfloat x,
                         GLfloat y,
                         GLfloat z)
{
    LOG_NP;
};

static void VertexAttrib3fv(PP_Resource context,
                          GLuint indx,
                          const GLfloat* values)
{
    LOG_NP;
};

static void VertexAttrib4f(PP_Resource context,
                         GLuint indx,
                         GLfloat x,
                         GLfloat y,
                         GLfloat z,
                         GLfloat w)
{
    LOG_NP;
};

static void VertexAttrib4fv(PP_Resource context,
                          GLuint indx,
                          const GLfloat* values)
{
    LOG_NP;
};

static void VertexAttribPointer(PP_Resource context,
                              GLuint indx,
                              GLint size,
                              GLenum type,
                              GLboolean normalized,
                              GLsizei stride,
                              const void* ptr)
{
    LOG_NP;
};

static void Viewport(PP_Resource context,
                   GLint x,
                   GLint y,
                   GLsizei width,
                   GLsizei height)
{
    LOG_NP;
};


struct PPB_OpenGLES2_1_0 PPB_OpenGLES2_1_0_instance =
{
    .ActiveTexture = ActiveTexture,
    .AttachShader = AttachShader,
    .BindAttribLocation = BindAttribLocation,
    .BindBuffer = BindBuffer,
    .BindFramebuffer = BindFramebuffer,
    .BindRenderbuffer = BindRenderbuffer,
    .BindTexture = BindTexture,
    .BlendColor = BlendColor,
    .BlendEquation = BlendEquation,
    .BlendEquationSeparate = BlendEquationSeparate,
    .BlendFunc = BlendFunc,
    .BlendFuncSeparate = BlendFuncSeparate,
    .BufferData = BufferData,
    .BufferSubData = BufferSubData,
    .CheckFramebufferStatus = CheckFramebufferStatus,
    .Clear = Clear,
    .ClearColor = ClearColor,
    .ClearDepthf = ClearDepthf,
    .ClearStencil = ClearStencil,
    .ColorMask = ColorMask,
    .CompileShader = CompileShader,
    .CompressedTexImage2D = CompressedTexImage2D,
    .CompressedTexSubImage2D = CompressedTexSubImage2D,
    .CopyTexImage2D = CopyTexImage2D,
    .CopyTexSubImage2D = CopyTexSubImage2D,
    .CreateProgram = CreateProgram,
    .CreateShader = CreateShader,
    .CullFace = CullFace,
    .DeleteBuffers = DeleteBuffers,
    .DeleteFramebuffers = DeleteFramebuffers,
    .DeleteProgram = DeleteProgram,
    .DeleteRenderbuffers = DeleteRenderbuffers,
    .DeleteShader = DeleteShader,
    .DeleteTextures = DeleteTextures,
    .DepthFunc = DepthFunc,
    .DepthMask = DepthMask,
    .DepthRangef = DepthRangef,
    .DetachShader = DetachShader,
    .Disable = Disable,
    .DisableVertexAttribArray = DisableVertexAttribArray,
    .DrawArrays = DrawArrays,
    .DrawElements = DrawElements,
    .Enable = Enable,
    .EnableVertexAttribArray = EnableVertexAttribArray,
    .Finish = Finish,
    .Flush = Flush,
    .FramebufferRenderbuffer = FramebufferRenderbuffer,
    .FramebufferTexture2D = FramebufferTexture2D,
    .FrontFace = FrontFace,
    .GenBuffers = GenBuffers,
    .GenerateMipmap = GenerateMipmap,
    .GenFramebuffers = GenFramebuffers,
    .GenRenderbuffers = GenRenderbuffers,
    .GenTextures = GenTextures,
    .GetActiveAttrib = GetActiveAttrib,
    .GetActiveUniform = GetActiveUniform,
    .GetAttachedShaders = GetAttachedShaders,
    .GetAttribLocation = GetAttribLocation,
    .GetBooleanv = GetBooleanv,
    .GetBufferParameteriv = GetBufferParameteriv,
    .GetError = GetError,
    .GetFloatv = GetFloatv,
    .GetFramebufferAttachmentParameteriv = GetFramebufferAttachmentParameteriv,
    .GetIntegerv = GetIntegerv,
    .GetProgramiv = GetProgramiv,
    .GetProgramInfoLog = GetProgramInfoLog,
    .GetRenderbufferParameteriv = GetRenderbufferParameteriv,
    .GetShaderiv = GetShaderiv,
    .GetShaderInfoLog = GetShaderInfoLog,
    .GetShaderPrecisionFormat = GetShaderPrecisionFormat,
    .GetShaderSource = GetShaderSource,
    .GetString = GetString,
    .GetTexParameterfv = GetTexParameterfv,
    .GetTexParameteriv = GetTexParameteriv,
    .GetUniformfv = GetUniformfv,
    .GetUniformiv = GetUniformiv,
    .GetUniformLocation = GetUniformLocation,
    .GetVertexAttribfv = GetVertexAttribfv,
    .GetVertexAttribiv = GetVertexAttribiv,
    .GetVertexAttribPointerv = GetVertexAttribPointerv,
    .Hint = Hint,
    .IsBuffer = IsBuffer,
    .IsEnabled = IsEnabled,
    .IsFramebuffer = IsFramebuffer,
    .IsProgram = IsProgram,
    .IsRenderbuffer = IsRenderbuffer,
    .IsShader = IsShader,
    .IsTexture = IsTexture,
    .LineWidth = LineWidth,
    .LinkProgram = LinkProgram,
    .PixelStorei = PixelStorei,
    .PolygonOffset = PolygonOffset,
    .ReadPixels = ReadPixels,
    .ReleaseShaderCompiler = ReleaseShaderCompiler,
    .RenderbufferStorage = RenderbufferStorage,
    .SampleCoverage = SampleCoverage,
    .Scissor = Scissor,
    .ShaderBinary = ShaderBinary,
    .ShaderSource = ShaderSource,
    .StencilFunc = StencilFunc,
    .StencilFuncSeparate = StencilFuncSeparate,
    .StencilMask = StencilMask,
    .StencilMaskSeparate = StencilMaskSeparate,
    .StencilOp = StencilOp,
    .StencilOpSeparate = StencilOpSeparate,
    .TexImage2D = TexImage2D,
    .TexParameterf = TexParameterf,
    .TexParameterfv = TexParameterfv,
    .TexParameteri = TexParameteri,
    .TexParameteriv = TexParameteriv,
    .TexSubImage2D = TexSubImage2D,
    .Uniform1f = Uniform1f,
    .Uniform1fv = Uniform1fv,
    .Uniform1i = Uniform1i,
    .Uniform1iv = Uniform1iv,
    .Uniform2f = Uniform2f,
    .Uniform2fv = Uniform2fv,
    .Uniform2i = Uniform2i,
    .Uniform2iv = Uniform2iv,
    .Uniform3f = Uniform3f,
    .Uniform3fv = Uniform3fv,
    .Uniform3i = Uniform3i,
    .Uniform3iv = Uniform3iv,
    .Uniform4f = Uniform4f,
    .Uniform4fv = Uniform4fv,
    .Uniform4i = Uniform4i,
    .Uniform4iv = Uniform4iv,
    .UniformMatrix2fv = UniformMatrix2fv,
    .UniformMatrix3fv = UniformMatrix3fv,
    .UniformMatrix4fv = UniformMatrix4fv,
    .UseProgram = UseProgram,
    .ValidateProgram = ValidateProgram,
    .VertexAttrib1f = VertexAttrib1f,
    .VertexAttrib1fv = VertexAttrib1fv,
    .VertexAttrib2f = VertexAttrib2f,
    .VertexAttrib2fv = VertexAttrib2fv,
    .VertexAttrib3f = VertexAttrib3f,
    .VertexAttrib3fv = VertexAttrib3fv,
    .VertexAttrib4f = VertexAttrib4f,
    .VertexAttrib4fv = VertexAttrib4fv,
    .VertexAttribPointer = VertexAttribPointer,
    .Viewport = Viewport,
};
