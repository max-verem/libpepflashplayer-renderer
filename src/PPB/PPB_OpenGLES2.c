#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_opengles2.h>

#include "log.h"

#include "PPB_Graphics3D.h"

static void ActiveTexture(PP_Resource context, GLenum texture)
{
    LOG_TD;
    glActiveTexture(texture);
};

static void AttachShader(PP_Resource context, GLuint program, GLuint shader)
{
    LOG_TD;
    glAttachShader(program, shader);
};

static void BindAttribLocation(PP_Resource context,
                             GLuint program,
                             GLuint index,
                             const char* name)
{
    LOG_TD;
    glBindAttribLocation(program, index, name);
};

static void BindBuffer(PP_Resource context, GLenum target, GLuint buffer)
{
    LOG_TD;
    glBindBuffer(target, buffer);
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
    LOG_TD;
    glBindTexture(target, texture);
};

static void BlendColor(PP_Resource context,
                     GLclampf red,
                     GLclampf green,
                     GLclampf blue,
                     GLclampf alpha)
{
    LOG_TD;
    glBlendColor(red, green, blue, alpha);
};

static void BlendEquation(PP_Resource context, GLenum mode)
{
    LOG_TD;
    glBlendEquation(mode);
};

static void BlendEquationSeparate(PP_Resource context,
                                GLenum modeRGB,
                                GLenum modeAlpha)
{
    LOG_TD;
    glBlendEquationSeparate(modeRGB, modeAlpha);
};

static void BlendFunc(PP_Resource context, GLenum sfactor, GLenum dfactor)
{
    LOG_TD;
    glBlendFunc(sfactor, dfactor);
};

static void BlendFuncSeparate(PP_Resource context,
                            GLenum srcRGB,
                            GLenum dstRGB,
                            GLenum srcAlpha,
                            GLenum dstAlpha)
{
    LOG_TD;
    glBlendFuncSeparate(srcRGB, dstRGB, srcAlpha, dstAlpha);
};

static void BufferData(PP_Resource context,
                     GLenum target,
                     GLsizeiptr size,
                     const void* data,
                     GLenum usage)
{
    LOG_TD;
    glBufferData(target, size, data, usage);
};

static void BufferSubData(PP_Resource context,
                        GLenum target,
                        GLintptr offset,
                        GLsizeiptr size,
                        const void* data)
{
    LOG_TD;
    glBufferSubData(target, offset, size, data);
};

static GLenum CheckFramebufferStatus(PP_Resource context, GLenum target)
{
    LOG_TD;
    return glCheckFramebufferStatus(target);
};

static void Clear(PP_Resource context, GLbitfield mask)
{
    LOG_TD;
    glClear(mask);
};

static void ClearColor(PP_Resource context,
                     GLclampf red,
                     GLclampf green,
                     GLclampf blue,
                     GLclampf alpha)
{
    LOG_TD;
    glClearColor(red, green, blue, alpha);
};

static void ClearDepthf(PP_Resource context, GLclampf depth)
{
    LOG_TD;
    glClearDepthf(depth);
};

static void ClearStencil(PP_Resource context, GLint s)
{
    LOG_TD;
    glClearDepthf(s);
};

static void ColorMask(PP_Resource context,
                    GLboolean red,
                    GLboolean green,
                    GLboolean blue,
                    GLboolean alpha)
{
    LOG_TD;
    glColorMask(red, green, blue, alpha);
};

static void CompileShader(PP_Resource context, GLuint shader)
{
    LOG_TD;
    glCompileShader(shader);
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
    LOG_TD;
    glCompressedTexImage2D(target, level, internalformat, width, height, border, imageSize, data);
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
    LOG_TD;
    glCompressedTexSubImage2D(target, level, xoffset, yoffset, width, height, format, imageSize, data);
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
    LOG_TD;
    glCopyTexImage2D(target, level, internalformat, x, y, width, height, border);
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
    LOG_TD;
    glCopyTexSubImage2D(target, level, xoffset, yoffset, x, y, width, height);
};

static GLuint CreateProgram(PP_Resource context)
{
    LOG_TD;
    return glCreateProgram();
};

static GLuint CreateShader(PP_Resource context, GLenum type)
{
    LOG_TD;
    return glCreateShader(type);
};

static void CullFace(PP_Resource context, GLenum mode)
{
    LOG_TD;
    glCullFace(mode);
};

static void DeleteBuffers(PP_Resource context, GLsizei n, const GLuint* buffers)
{
    LOG_TD;
    glDeleteBuffers(n, buffers);
};

static void DeleteFramebuffers(PP_Resource context,
                             GLsizei n,
                             const GLuint* framebuffers)
{
    LOG_TD;
    glDeleteFramebuffers(n, framebuffers);
};

static void DeleteProgram(PP_Resource context, GLuint program)
{
    LOG_TD;
    glDeleteProgram(program);
};

static void DeleteRenderbuffers(PP_Resource context,
                              GLsizei n,
                              const GLuint* renderbuffers)
{
    LOG_TD;
    glDeleteRenderbuffers(n, renderbuffers);
};

static void DeleteShader(PP_Resource context, GLuint shader)
{
    LOG_TD;
    glDeleteShader(shader);
};

static void DeleteTextures(PP_Resource context,
                         GLsizei n,
                         const GLuint* textures)
{
    LOG_TD;
    glDeleteTextures(n, textures);
};

static void DepthFunc(PP_Resource context, GLenum func)
{
    LOG_TD;
    glDepthFunc(func);
};

static void DepthMask(PP_Resource context, GLboolean flag)
{
    LOG_TD;
    glDepthMask(flag);
};

static void DepthRangef(PP_Resource context, GLclampf zNear, GLclampf zFar)
{
    LOG_TD;
    glDepthRangef(zNear, zFar);
};

static void DetachShader(PP_Resource context, GLuint program, GLuint shader)
{
    LOG_TD;
    glDetachShader(program, shader);
};

static void Disable(PP_Resource context, GLenum cap)
{
    LOG_TD;
    glDisable(cap);
};

static void DisableVertexAttribArray(PP_Resource context, GLuint index)
{
    LOG_TD;
    glDisableVertexAttribArray(index);
};

static void DrawArrays(PP_Resource context,
                     GLenum mode,
                     GLint first,
                     GLsizei count)
{
    LOG_TD;
    glDrawArrays(mode, first, count);
};

static void DrawElements(PP_Resource context,
                       GLenum mode,
                       GLsizei count,
                       GLenum type,
                       const void* indices)
{
    LOG_TD;
    glDrawElements(mode, count, type, indices);
};

static void Enable(PP_Resource context, GLenum cap)
{
    LOG_TD;
    glEnable(cap);
};

static void EnableVertexAttribArray(PP_Resource context, GLuint index)
{
    LOG_TD;
    glEnableVertexAttribArray(index);
};

static void Finish(PP_Resource context)
{
    LOG_TD;
    glFinish();
};

static void Flush(PP_Resource context)
{
    LOG_TD;
    glFlush();
};

static void FramebufferRenderbuffer(PP_Resource context,
                                  GLenum target,
                                  GLenum attachment,
                                  GLenum renderbuffertarget,
                                  GLuint renderbuffer)
{
    LOG_TD;
    glFramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer);
};

static void FramebufferTexture2D(PP_Resource context,
                               GLenum target,
                               GLenum attachment,
                               GLenum textarget,
                               GLuint texture,
                               GLint level)
{
    LOG_TD;
    glFramebufferTexture2D(target, attachment, textarget, texture, level);
};

static void FrontFace(PP_Resource context, GLenum mode)
{
    LOG_TD;
    glFrontFace(mode);
};

static void GenBuffers(PP_Resource context, GLsizei n, GLuint* buffers)
{
    LOG_TD;
    glGenBuffers(n, buffers);
};

static void GenerateMipmap(PP_Resource context, GLenum target)
{
    LOG_TD;
    glGenerateMipmap(target);
};

static void GenFramebuffers(PP_Resource context, GLsizei n, GLuint* framebuffers)
{
    LOG_TD;
    glGenFramebuffers(n, framebuffers);
};

static void GenRenderbuffers(PP_Resource context,
                           GLsizei n,
                           GLuint* renderbuffers)
{
    LOG_TD;
    glGenRenderbuffers(n, renderbuffers);
};

static void GenTextures(PP_Resource context, GLsizei n, GLuint* textures)
{
    LOG_TD;
    glGenTextures(n, textures);
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
    LOG_TD;
    glGetActiveAttrib(program, index, bufsize, length, size, type, name);
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
    LOG_TD;
    glGetActiveUniform(program, index, bufsize, length, size, type, name);
};

static void GetAttachedShaders(PP_Resource context,
                             GLuint program,
                             GLsizei maxcount,
                             GLsizei* count,
                             GLuint* shaders)
{
    LOG_TD;
    glGetAttachedShaders(program, maxcount, count, shaders);
};

static GLint GetAttribLocation(PP_Resource context,
                             GLuint program,
                             const char* name)
{
    LOG_TD;
    return glGetAttribLocation(program, name);
};

static void GetBooleanv(PP_Resource context, GLenum pname, GLboolean* params)
{
    LOG_TD;
    glGetBooleanv(pname, params);
};

static void GetBufferParameteriv(PP_Resource context,
                               GLenum target,
                               GLenum pname,
                               GLint* params)
{
    LOG_TD;
    glGetBufferParameteriv(target, pname, params);
};

static GLenum GetError(PP_Resource context)
{
    LOG_TD;
    return glGetError();
};

static void GetFloatv(PP_Resource context, GLenum pname, GLfloat* params)
{
    LOG_TD;
    glGetFloatv(pname, params);
};

static void GetFramebufferAttachmentParameteriv(PP_Resource context,
                                              GLenum target,
                                              GLenum attachment,
                                              GLenum pname,
                                              GLint* params)
{
    LOG_TD;
    glGetFramebufferAttachmentParameteriv(target, attachment, pname, params);
};

static void GetIntegerv(PP_Resource context, GLenum pname, GLint* params)
{
    LOG_TD;
    glGetIntegerv(pname, params);
};

static void GetProgramiv(PP_Resource context,
                       GLuint program,
                       GLenum pname,
                       GLint* params)
{
    LOG_TD;
    glGetProgramiv(program, pname, params);
};

static void GetProgramInfoLog(PP_Resource context,
                            GLuint program,
                            GLsizei bufsize,
                            GLsizei* length,
                            char* infolog)
{
    LOG_TD;
    glGetProgramInfoLog(program, bufsize, length, infolog);
};

static void GetRenderbufferParameteriv(PP_Resource context,
                                     GLenum target,
                                     GLenum pname,
                                     GLint* params)
{
    LOG_TD;
    glGetRenderbufferParameteriv(target, pname, params);
};

static void GetShaderiv(PP_Resource context,
                      GLuint shader,
                      GLenum pname,
                      GLint* params)
{
    LOG_TD;
    glGetShaderiv(shader, pname, params);
};

static void GetShaderInfoLog(PP_Resource context,
                           GLuint shader,
                           GLsizei bufsize,
                           GLsizei* length,
                           char* infolog)
{
    LOG_TD;
    glGetShaderInfoLog(shader, bufsize, length, infolog);
};

static void GetShaderPrecisionFormat(PP_Resource context,
                                   GLenum shadertype,
                                   GLenum precisiontype,
                                   GLint* range,
                                   GLint* precision)
{
    LOG_TD;
    glGetShaderPrecisionFormat(shadertype, precisiontype, range, precision);
};

static void GetShaderSource(PP_Resource context,
                          GLuint shader,
                          GLsizei bufsize,
                          GLsizei* length,
                          char* source)
{
    LOG_TD;
    glGetShaderSource(shader, bufsize, length, source);
};

static const GLubyte* GetString(PP_Resource context, GLenum name)
{
    LOG_TD;
    return glGetString(name);
};

static void GetTexParameterfv(PP_Resource context,
                            GLenum target,
                            GLenum pname,
                            GLfloat* params)
{
    LOG_TD;
    glGetTexParameterfv(target, pname, params);
};

static void GetTexParameteriv(PP_Resource context,
                            GLenum target,
                            GLenum pname,
                            GLint* params)
{
    LOG_TD;
    glGetTexParameteriv(target, pname, params);
};

static void GetUniformfv(PP_Resource context,
                       GLuint program,
                       GLint location,
                       GLfloat* params)
{
    LOG_TD;
    glGetUniformfv(program, location, params);
};

static void GetUniformiv(PP_Resource context,
                       GLuint program,
                       GLint location,
                       GLint* params)
{
    LOG_TD;
    glGetUniformiv(program, location, params);
};

static GLint GetUniformLocation(PP_Resource context,
                              GLuint program,
                              const char* name)
{
    LOG_TD;
    return glGetUniformLocation(program, name);
};

static void GetVertexAttribfv(PP_Resource context,
                            GLuint index,
                            GLenum pname,
                            GLfloat* params)
{
    LOG_TD;
    glGetVertexAttribfv(index, pname, params);
};

static void GetVertexAttribiv(PP_Resource context,
                            GLuint index,
                            GLenum pname,
                            GLint* params)
{
    LOG_TD;
    glGetVertexAttribiv(index, pname, params);
};

static void GetVertexAttribPointerv(PP_Resource context,
                                  GLuint index,
                                  GLenum pname,
                                  void** pointer)
{
    LOG_TD;
    glGetVertexAttribPointerv(index, pname, pointer);
};

static void Hint(PP_Resource context, GLenum target, GLenum mode)
{
    LOG_TD;
    glHint(target, mode);
};

static GLboolean IsBuffer(PP_Resource context, GLuint buffer)
{
    LOG_TD;
    return glIsBuffer(buffer);
};

static GLboolean IsEnabled(PP_Resource context, GLenum cap)
{
    LOG_TD;
    return glIsEnabled(cap);
};

static GLboolean IsFramebuffer(PP_Resource context, GLuint framebuffer)
{
    LOG_TD;
    return glIsFramebuffer(framebuffer);
};

static GLboolean IsProgram(PP_Resource context, GLuint program)
{
    LOG_TD;
    return glIsProgram(program);
};

static GLboolean IsRenderbuffer(PP_Resource context, GLuint renderbuffer)
{
    LOG_TD;
    return glIsRenderbuffer(renderbuffer);
};

static GLboolean IsShader(PP_Resource context, GLuint shader)
{
    LOG_TD;
    return glIsShader(shader);
};

static GLboolean IsTexture(PP_Resource context, GLuint texture)
{
    LOG_TD;
    return glIsTexture(texture);
};

static void LineWidth(PP_Resource context, GLfloat width)
{
    LOG_TD;
    glLineWidth(width);
};

static void LinkProgram(PP_Resource context, GLuint program)
{
    LOG_TD;
    glLinkProgram(program);
};

static void PixelStorei(PP_Resource context, GLenum pname, GLint param)
{
    LOG_TD;
    glPixelStorei(pname, param);
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
    LOG_TD;
    glReleaseShaderCompiler();
};

static void RenderbufferStorage(PP_Resource context,
                              GLenum target,
                              GLenum internalformat,
                              GLsizei width,
                              GLsizei height)
{
    LOG_TD;
    glRenderbufferStorage(target, internalformat, width, height);
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
    LOG_TD;
    glShaderBinary(n, shaders, binaryformat, binary, length);
};

static void ShaderSource(PP_Resource context,
                       GLuint shader,
                       GLsizei count,
                       const char** str,
                       const GLint* length)
{
    LOG_TD;
    glShaderSource(shader, count, str, length);
};

static void StencilFunc(PP_Resource context, GLenum func, GLint ref, GLuint mask)
{
    LOG_TD;
    glStencilFunc(func, ref, mask);
};

static void StencilFuncSeparate(PP_Resource context,
                              GLenum face,
                              GLenum func,
                              GLint ref,
                              GLuint mask)
{
    LOG_TD;
    glStencilFuncSeparate(face, func, ref, mask);
};

static void StencilMask(PP_Resource context, GLuint mask)
{
    LOG_TD;
    glStencilMask(mask);
};

static void StencilMaskSeparate(PP_Resource context, GLenum face, GLuint mask)
{
    LOG_TD;
    glStencilMaskSeparate(face, mask);
};

static void StencilOp(PP_Resource context,
                    GLenum fail,
                    GLenum zfail,
                    GLenum zpass)
{
    LOG_TD;
    glStencilOp(fail, zfail, zpass);
};

static void StencilOpSeparate(PP_Resource context,
                            GLenum face,
                            GLenum fail,
                            GLenum zfail,
                            GLenum zpass)
{
    LOG_TD;
    glStencilOpSeparate(face, fail, zfail, zpass);
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
    LOG_TD;
    glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels);
};

static void TexParameterf(PP_Resource context,
                        GLenum target,
                        GLenum pname,
                        GLfloat param)
{
    LOG_TD;
    glTexParameterf(target, pname, param);
};

static void TexParameterfv(PP_Resource context,
                         GLenum target,
                         GLenum pname,
                         const GLfloat* params)
{
    LOG_TD;
    glTexParameterfv(target, pname, params);
};

static void TexParameteri(PP_Resource context,
                        GLenum target,
                        GLenum pname,
                        GLint param)
{
    LOG_TD;
    glTexParameteri(target, pname, param);
};

static void TexParameteriv(PP_Resource context,
                         GLenum target,
                         GLenum pname,
                         const GLint* params)
{
    LOG_TD;
    glTexParameteriv(target, pname, params);
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
    LOG_TD;
    glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels);
};

static void Uniform1f(PP_Resource context, GLint location, GLfloat x)
{
    LOG_TD;
    glUniform1f(location, x);
};

static void Uniform1fv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLfloat* v)
{
    LOG_TD;
    glUniform1fv(location, count, v);
};

static void Uniform1i(PP_Resource context, GLint location, GLint x)
{
    LOG_TD;
    glUniform1i(location, x);
};

static void Uniform1iv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLint* v)
{
    LOG_TD;
    glUniform1iv(location, count, v);
};

static void Uniform2f(PP_Resource context, GLint location, GLfloat x, GLfloat y)
{
    LOG_TD;
    glUniform2f(location, x, y);
};

static  void Uniform2fv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLfloat* v)
{
    LOG_TD;
    glUniform2fv(location, count, v);
};

static void Uniform2i(PP_Resource context, GLint location, GLint x, GLint y)
{
    LOG_TD;
    glUniform2i(location, x, y);
};

static void Uniform2iv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLint* v)
{
    LOG_TD;
    glUniform2iv(location, count, v);
};

static void Uniform3f(PP_Resource context,
                    GLint location,
                    GLfloat x,
                    GLfloat y,
                    GLfloat z)
{
    LOG_TD;
    glUniform3f(location, x, y, z);
};

static void Uniform3fv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLfloat* v)
{
    LOG_TD;
    glUniform3fv(location, count, v);
};

static void Uniform3i(PP_Resource context,
                    GLint location,
                    GLint x,
                    GLint y,
                    GLint z)
{
    LOG_TD;
    glUniform3i(location, x, y, z);
};

static void Uniform3iv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLint* v)
{
    LOG_TD;
    glUniform3iv(location, count, v);
};

static void Uniform4f(PP_Resource context,
                    GLint location,
                    GLfloat x,
                    GLfloat y,
                    GLfloat z,
                    GLfloat w)
{
    LOG_TD;
    glUniform4f(location, x, y, z, w);
};

static void Uniform4fv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLfloat* v)
{
    LOG_TD;
    glUniform4fv(location, count, v);
};

static void Uniform4i(PP_Resource context,
                    GLint location,
                    GLint x,
                    GLint y,
                    GLint z,
                    GLint w)
{
    LOG_TD;
    glUniform4i(location, x, y, z, w);
};

static void Uniform4iv(PP_Resource context,
                     GLint location,
                     GLsizei count,
                     const GLint* v)
{
    LOG_TD;
    glUniform4iv(location, count, v);
};

static void UniformMatrix2fv(PP_Resource context,
                           GLint location,
                           GLsizei count,
                           GLboolean transpose,
                           const GLfloat* value)
{
    LOG_TD;
    glUniformMatrix2fv(location, count, transpose, value);
};

static void UniformMatrix3fv(PP_Resource context,
                           GLint location,
                           GLsizei count,
                           GLboolean transpose,
                           const GLfloat* value)
{
    LOG_TD;
    glUniformMatrix3fv(location, count, transpose, value);
};

static void UniformMatrix4fv(PP_Resource context,
                           GLint location,
                           GLsizei count,
                           GLboolean transpose,
                           const GLfloat* value)
{
    LOG_TD;
    glUniformMatrix4fv(location, count, transpose, value);
};

static void UseProgram(PP_Resource context, GLuint program)
{
    LOG_TD;
    glUseProgram(program);
};

static void ValidateProgram(PP_Resource context, GLuint program)
{
    LOG_TD;
    glValidateProgram(program);
};

static void VertexAttrib1f(PP_Resource context, GLuint indx, GLfloat x)
{
    LOG_TD;
    glVertexAttrib1f(indx, x);
};

static void VertexAttrib1fv(PP_Resource context,
                          GLuint indx,
                          const GLfloat* values)
{
    LOG_TD;
    glVertexAttrib1fv(indx, values);
};

static void VertexAttrib2f(PP_Resource context,
                         GLuint indx,
                         GLfloat x,
                         GLfloat y)
{
    LOG_TD;
    glVertexAttrib2f(indx, x, y);
};

static void VertexAttrib2fv(PP_Resource context,
                          GLuint indx,
                          const GLfloat* values)
{
    LOG_TD;
    glVertexAttrib2fv(indx, values);
};

static void VertexAttrib3f(PP_Resource context,
                         GLuint indx,
                         GLfloat x,
                         GLfloat y,
                         GLfloat z)
{
    LOG_TD;
    glVertexAttrib3f(indx, x, y, z);
};

static void VertexAttrib3fv(PP_Resource context,
                          GLuint indx,
                          const GLfloat* values)
{
    LOG_TD;
    glVertexAttrib3fv(indx, values);
};

static void VertexAttrib4f(PP_Resource context,
                         GLuint indx,
                         GLfloat x,
                         GLfloat y,
                         GLfloat z,
                         GLfloat w)
{
    LOG_TD;
    glVertexAttrib4f(indx, x, y, z, w);
};

static void VertexAttrib4fv(PP_Resource context,
                          GLuint indx,
                          const GLfloat* values)
{
    LOG_TD;
    glVertexAttrib4fv(indx, values);
};

static void VertexAttribPointer(PP_Resource context,
                              GLuint indx,
                              GLint size,
                              GLenum type,
                              GLboolean normalized,
                              GLsizei stride,
                              const void* ptr)
{
    LOG_TD;
    glVertexAttribPointer(indx, size, type, normalized, stride, ptr);
};

static void Viewport(PP_Resource context,
                   GLint x,
                   GLint y,
                   GLsizei width,
                   GLsizei height)
{
    LOG_TD;
    glViewport(x, y, width, height);
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
