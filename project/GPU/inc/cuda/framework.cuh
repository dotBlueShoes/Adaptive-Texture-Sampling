#pragma once
//
#include <blue/error.hpp>
//
#include <cudagl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>


//#define CUDA_GET_ERROR(errorCode, msg) { \
//    if (errorCode != cudaError::cudaSuccess) { \
//        const auto&& errorStr = cudaGetErrorString (errorCode); \
//        LOGERROR ("CUDA: [%d-%s]" " at (%s)\n", errorCode, errorStr, msg); \
//    } \
//}

#define THREADS_16 16


#define CUDA_GET_ERROR(errorCode, msg) { \
    switch (errorCode) { \
        case cudaError::cudaSuccess:                                {} break; \
        case cudaError::cudaErrorInvalidValue:                      { LOGERROR ("CUDA: [%d-cudaErrorInvalidValue]"                      " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorMemoryAllocation:                  { LOGERROR ("CUDA: [%d-cudaErrorMemoryAllocation]"                  " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInitializationError:               { LOGERROR ("CUDA: [%d-cudaErrorInitializationError]"               " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorCudartUnloading:                   { LOGERROR ("CUDA: [%d-cudaErrorCudartUnloading]"                   " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorProfilerDisabled:                  { LOGERROR ("CUDA: [%d-cudaErrorProfilerDisabled]"                  " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorProfilerNotInitialized:            { LOGERROR ("CUDA: [%d-cudaErrorProfilerNotInitialized]"            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorProfilerAlreadyStarted:            { LOGERROR ("CUDA: [%d-cudaErrorProfilerAlreadyStarted]"            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorProfilerAlreadyStopped:            { LOGERROR ("CUDA: [%d-cudaErrorProfilerAlreadyStopped]"            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidConfiguration:              { LOGERROR ("CUDA: [%d-cudaErrorInvalidConfiguration]"              " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidPitchValue:                 { LOGERROR ("CUDA: [%d-cudaErrorInvalidPitchValue]"                 " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidSymbol:                     { LOGERROR ("CUDA: [%d-cudaErrorInvalidSymbol]"                     " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidHostPointer:                { LOGERROR ("CUDA: [%d-cudaErrorInvalidHostPointer]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidDevicePointer:              { LOGERROR ("CUDA: [%d-cudaErrorInvalidDevicePointer]"              " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidTexture:                    { LOGERROR ("CUDA: [%d-cudaErrorInvalidTexture]"                    " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidTextureBinding:             { LOGERROR ("CUDA: [%d-cudaErrorInvalidTextureBinding]"             " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidChannelDescriptor:          { LOGERROR ("CUDA: [%d-cudaErrorInvalidChannelDescriptor]"          " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidMemcpyDirection:            { LOGERROR ("CUDA: [%d-cudaErrorInvalidMemcpyDirection]"            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorAddressOfConstant:                 { LOGERROR ("CUDA: [%d-cudaErrorAddressOfConstant]"                 " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorTextureFetchFailed:                { LOGERROR ("CUDA: [%d-cudaErrorTextureFetchFailed]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorTextureNotBound:                   { LOGERROR ("CUDA: [%d-cudaErrorTextureNotBound]"                   " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorSynchronizationError:              { LOGERROR ("CUDA: [%d-cudaErrorSynchronizationError]"              " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidFilterSetting:              { LOGERROR ("CUDA: [%d-cudaErrorInvalidFilterSetting]"              " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidNormSetting:                { LOGERROR ("CUDA: [%d-cudaErrorInvalidNormSetting]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorMixedDeviceExecution:              { LOGERROR ("CUDA: [%d-cudaErrorMixedDeviceExecution]"              " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorNotYetImplemented:                 { LOGERROR ("CUDA: [%d-cudaErrorNotYetImplemented]"                 " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorMemoryValueTooLarge:               { LOGERROR ("CUDA: [%d-cudaErrorMemoryValueTooLarge]"               " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorStubLibrary:                       { LOGERROR ("CUDA: [%d-cudaErrorStubLibrary]"                       " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInsufficientDriver:                { LOGERROR ("CUDA: [%d-cudaErrorInsufficientDriver]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorCallRequiresNewerDriver:           { LOGERROR ("CUDA: [%d-cudaErrorCallRequiresNewerDriver]"           " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidSurface:                    { LOGERROR ("CUDA: [%d-cudaErrorInvalidSurface]"                    " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorDuplicateVariableName:             { LOGERROR ("CUDA: [%d-cudaErrorDuplicateVariableName]"             " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorDuplicateTextureName:              { LOGERROR ("CUDA: [%d-cudaErrorDuplicateTextureName]"              " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorDuplicateSurfaceName:              { LOGERROR ("CUDA: [%d-cudaErrorDuplicateSurfaceName]"              " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorDevicesUnavailable:                { LOGERROR ("CUDA: [%d-cudaErrorDevicesUnavailable]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorIncompatibleDriverContext:         { LOGERROR ("CUDA: [%d-cudaErrorIncompatibleDriverContext]"         " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorMissingConfiguration:              { LOGERROR ("CUDA: [%d-cudaErrorMissingConfiguration]"              " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorPriorLaunchFailure:                { LOGERROR ("CUDA: [%d-cudaErrorPriorLaunchFailure]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorLaunchMaxDepthExceeded:            { LOGERROR ("CUDA: [%d-cudaErrorLaunchMaxDepthExceeded]"            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorLaunchFileScopedTex:               { LOGERROR ("CUDA: [%d-cudaErrorLaunchFileScopedTex]"               " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorLaunchFileScopedSurf:              { LOGERROR ("CUDA: [%d-cudaErrorLaunchFileScopedSurf]"              " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorSyncDepthExceeded:                 { LOGERROR ("CUDA: [%d-cudaErrorSyncDepthExceeded]"                 " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorLaunchPendingCountExceeded:        { LOGERROR ("CUDA: [%d-cudaErrorLaunchPendingCountExceeded]"        " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidDeviceFunction:             { LOGERROR ("CUDA: [%d-cudaErrorInvalidDeviceFunction]"             " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorNoDevice:                          { LOGERROR ("CUDA: [%d-cudaErrorNoDevice]"                          " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidDevice:                     { LOGERROR ("CUDA: [%d-cudaErrorInvalidDevice]"                     " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorDeviceNotLicensed:                 { LOGERROR ("CUDA: [%d-cudaErrorDeviceNotLicensed]"                 " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorSoftwareValidityNotEstablished:    { LOGERROR ("CUDA: [%d-cudaErrorSoftwareValidityNotEstablished]"    " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorStartupFailure:                    { LOGERROR ("CUDA: [%d-cudaErrorStartupFailure]"                    " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidKernelImage:                { LOGERROR ("CUDA: [%d-cudaErrorInvalidKernelImage]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorDeviceUninitialized:               { LOGERROR ("CUDA: [%d-cudaErrorDeviceUninitialized]"               " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorMapBufferObjectFailed:             { LOGERROR ("CUDA: [%d-cudaErrorMapBufferObjectFailed]"             " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorUnmapBufferObjectFailed:           { LOGERROR ("CUDA: [%d-cudaErrorUnmapBufferObjectFailed]"           " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorArrayIsMapped:                     { LOGERROR ("CUDA: [%d-cudaErrorArrayIsMapped]"                     " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorAlreadyMapped:                     { LOGERROR ("CUDA: [%d-cudaErrorAlreadyMapped]"                     " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorNoKernelImageForDevice:            { LOGERROR ("CUDA: [%d-cudaErrorNoKernelImageForDevice]"            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorAlreadyAcquired:                   { LOGERROR ("CUDA: [%d-cudaErrorAlreadyAcquired]"                   " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorNotMapped:                         { LOGERROR ("CUDA: [%d-cudaErrorNotMapped]"                         " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorNotMappedAsArray:                  { LOGERROR ("CUDA: [%d-cudaErrorNotMappedAsArray]"                  " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorNotMappedAsPointer:                { LOGERROR ("CUDA: [%d-cudaErrorNotMappedAsPointer]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorECCUncorrectable:                  { LOGERROR ("CUDA: [%d-cudaErrorECCUncorrectable]"                  " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorUnsupportedLimit:                  { LOGERROR ("CUDA: [%d-cudaErrorUnsupportedLimit]"                  " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorDeviceAlreadyInUse:                { LOGERROR ("CUDA: [%d-cudaErrorDeviceAlreadyInUse]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorPeerAccessUnsupported:             { LOGERROR ("CUDA: [%d-cudaErrorPeerAccessUnsupported]"             " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidPtx:                        { LOGERROR ("CUDA: [%d-cudaErrorInvalidPtx]"                        " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidGraphicsContext:            { LOGERROR ("CUDA: [%d-cudaErrorInvalidGraphicsContext]"            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorNvlinkUncorrectable:               { LOGERROR ("CUDA: [%d-cudaErrorNvlinkUncorrectable]"               " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorJitCompilerNotFound:               { LOGERROR ("CUDA: [%d-cudaErrorJitCompilerNotFound]"               " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorUnsupportedPtxVersion:             { LOGERROR ("CUDA: [%d-cudaErrorUnsupportedPtxVersion]"             " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorJitCompilationDisabled:            { LOGERROR ("CUDA: [%d-cudaErrorJitCompilationDisabled]"            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorUnsupportedExecAffinity:           { LOGERROR ("CUDA: [%d-cudaErrorUnsupportedExecAffinity]"           " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorUnsupportedDevSideSync:            { LOGERROR ("CUDA: [%d-cudaErrorUnsupportedDevSideSync]"            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidSource:                     { LOGERROR ("CUDA: [%d-cudaErrorInvalidSource]"                     " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorFileNotFound:                      { LOGERROR ("CUDA: [%d-cudaErrorFileNotFound]"                      " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorSharedObjectSymbolNotFound:        { LOGERROR ("CUDA: [%d-cudaErrorSharedObjectSymbolNotFound]"        " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorSharedObjectInitFailed:            { LOGERROR ("CUDA: [%d-cudaErrorSharedObjectInitFailed]"            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorOperatingSystem:                   { LOGERROR ("CUDA: [%d-cudaErrorOperatingSystem]"                   " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidResourceHandle:             { LOGERROR ("CUDA: [%d-cudaErrorInvalidResourceHandle]"             " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorIllegalState:                      { LOGERROR ("CUDA: [%d-cudaErrorIllegalState]"                      " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorLossyQuery:                        { LOGERROR ("CUDA: [%d-cudaErrorLossyQuery]"                        " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorSymbolNotFound:                    { LOGERROR ("CUDA: [%d-cudaErrorSymbolNotFound]"                    " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorNotReady:                          { LOGERROR ("CUDA: [%d-cudaErrorNotReady]"                          " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorIllegalAddress:                    { LOGERROR ("CUDA: [%d-cudaErrorIllegalAddress]"                    " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorLaunchOutOfResources:              { LOGERROR ("CUDA: [%d-cudaErrorLaunchOutOfResources]"              " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorLaunchTimeout:                     { LOGERROR ("CUDA: [%d-cudaErrorLaunchTimeout]"                     " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorLaunchIncompatibleTexturing:       { LOGERROR ("CUDA: [%d-cudaErrorLaunchIncompatibleTexturing]"       " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorPeerAccessAlreadyEnabled:          { LOGERROR ("CUDA: [%d-cudaErrorPeerAccessAlreadyEnabled]"          " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorPeerAccessNotEnabled:              { LOGERROR ("CUDA: [%d-cudaErrorPeerAccessNotEnabled]"              " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorSetOnActiveProcess:                { LOGERROR ("CUDA: [%d-cudaErrorSetOnActiveProcess]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorContextIsDestroyed:                { LOGERROR ("CUDA: [%d-cudaErrorContextIsDestroyed]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorAssert:                            { LOGERROR ("CUDA: [%d-cudaErrorAssert]"                            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorTooManyPeers:                      { LOGERROR ("CUDA: [%d-cudaErrorTooManyPeers]"                      " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorHostMemoryAlreadyRegistered:       { LOGERROR ("CUDA: [%d-cudaErrorHostMemoryAlreadyRegistered]"       " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorHostMemoryNotRegistered:           { LOGERROR ("CUDA: [%d-cudaErrorHostMemoryNotRegistered]"           " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorHardwareStackError:                { LOGERROR ("CUDA: [%d-cudaErrorHardwareStackError]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorIllegalInstruction:                { LOGERROR ("CUDA: [%d-cudaErrorIllegalInstruction]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorMisalignedAddress:                 { LOGERROR ("CUDA: [%d-cudaErrorMisalignedAddress]"                 " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidAddressSpace:               { LOGERROR ("CUDA: [%d-cudaErrorInvalidAddressSpace]"               " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidPc:                         { LOGERROR ("CUDA: [%d-cudaErrorInvalidPc]"                         " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorLaunchFailure:                     { LOGERROR ("CUDA: [%d-cudaErrorLaunchFailure]"                     " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorCooperativeLaunchTooLarge:         { LOGERROR ("CUDA: [%d-cudaErrorCooperativeLaunchTooLarge]"         " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorNotPermitted:                      { LOGERROR ("CUDA: [%d-cudaErrorNotPermitted]"                      " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorNotSupported:                      { LOGERROR ("CUDA: [%d-cudaErrorNotSupported]"                      " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorSystemNotReady:                    { LOGERROR ("CUDA: [%d-cudaErrorSystemNotReady]"                    " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorSystemDriverMismatch:              { LOGERROR ("CUDA: [%d-cudaErrorSystemDriverMismatch]"              " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorCompatNotSupportedOnDevice:        { LOGERROR ("CUDA: [%d-cudaErrorCompatNotSupportedOnDevice]"        " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorMpsConnectionFailed:               { LOGERROR ("CUDA: [%d-cudaErrorMpsConnectionFailed]"               " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorMpsRpcFailure:                     { LOGERROR ("CUDA: [%d-cudaErrorMpsRpcFailure]"                     " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorMpsServerNotReady:                 { LOGERROR ("CUDA: [%d-cudaErrorMpsServerNotReady]"                 " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorMpsMaxClientsReached:              { LOGERROR ("CUDA: [%d-cudaErrorMpsMaxClientsReached]"              " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorMpsMaxConnectionsReached:          { LOGERROR ("CUDA: [%d-cudaErrorMpsMaxConnectionsReached]"          " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorMpsClientTerminated:               { LOGERROR ("CUDA: [%d-cudaErrorMpsClientTerminated]"               " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorCdpNotSupported:                   { LOGERROR ("CUDA: [%d-cudaErrorCdpNotSupported]"                   " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorCdpVersionMismatch:                { LOGERROR ("CUDA: [%d-cudaErrorCdpVersionMismatch]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorStreamCaptureUnsupported:          { LOGERROR ("CUDA: [%d-cudaErrorStreamCaptureUnsupported]"          " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorStreamCaptureInvalidated:          { LOGERROR ("CUDA: [%d-cudaErrorStreamCaptureInvalidated]"          " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorStreamCaptureMerge:                { LOGERROR ("CUDA: [%d-cudaErrorStreamCaptureMerge]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorStreamCaptureUnmatched:            { LOGERROR ("CUDA: [%d-cudaErrorStreamCaptureUnmatched]"            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorStreamCaptureUnjoined:             { LOGERROR ("CUDA: [%d-cudaErrorStreamCaptureUnjoined]"             " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorStreamCaptureIsolation:            { LOGERROR ("CUDA: [%d-cudaErrorStreamCaptureIsolation]"            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorStreamCaptureImplicit:             { LOGERROR ("CUDA: [%d-cudaErrorStreamCaptureImplicit]"             " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorCapturedEvent:                     { LOGERROR ("CUDA: [%d-cudaErrorCapturedEvent]"                     " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorStreamCaptureWrongThread:          { LOGERROR ("CUDA: [%d-cudaErrorStreamCaptureWrongThread]"          " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorTimeout:                           { LOGERROR ("CUDA: [%d-cudaErrorTimeout]"                           " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorGraphExecUpdateFailure:            { LOGERROR ("CUDA: [%d-cudaErrorGraphExecUpdateFailure]"            " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorExternalDevice:                    { LOGERROR ("CUDA: [%d-cudaErrorExternalDevice]"                    " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidClusterSize:                { LOGERROR ("CUDA: [%d-cudaErrorInvalidClusterSize]"                " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorFunctionNotLoaded:                 { LOGERROR ("CUDA: [%d-cudaErrorFunctionNotLoaded]"                 " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidResourceType:               { LOGERROR ("CUDA: [%d-cudaErrorInvalidResourceType]"               " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorInvalidResourceConfiguration:      { LOGERROR ("CUDA: [%d-cudaErrorInvalidResourceConfiguration]"      " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorUnknown:                           { LOGERROR ("CUDA: [%d-cudaErrorUnknown]"                           " at (%s)\n", errorCode, msg); } break; \
        case cudaError::cudaErrorApiFailureBase:                    { LOGERROR ("CUDA: [%d-cudaErrorApiFailureBase]"                    " at (%s)\n", errorCode, msg); } break; \
        default:                                                    { LOGERROR ("CUDA: [%d-atsErrorUnknown]"                            " at (%s)\n", errorCode, msg); } \
    } \
}


// `cudaGetLastError`	    -> After every kernel launch or async API call. Detect immediate launch failures. Returns last error, clears it.
// `cudaPeekAtLastError`	-> For non-destructive peeking, mostly for debugging. Returns last error without clearing it.
// `cudaDeviceSynchronize`	-> When you want to wait for all GPU work to complete and catch runtime errors.	Returns any async errors from device-side execution.

#ifdef ATS_ENABLE_DEEP_DEBUG

    // NOTE. If all GPU work is done the error might be propagated with `cudaGetLastError` call.
    // TODO. I should be calling `ERROR` instead of `LOGERROR`.

    #define KERNEL_GET_ERROR(msg) { \
        cudaError_t _error = cudaGetLastError(); \
        if (_error != cudaSuccess) { \
            LOGERROR ("CUDA Kernel (%s) [Launch] ERROR : %s\n", msg, cudaGetErrorString (_error)); \
        } \
        _error = cudaDeviceSynchronize (); \
        if (_error != cudaSuccess) { \
            LOGERROR ("CUDA Kernel (%s) [Execution] ERROR : %s\n", msg, cudaGetErrorString (_error)); \
        } \
    }

#else

    #define KERNEL_GET_ERROR(msg) { \
        cudaError_t _error = cudaGetLastError (); \
        if (_error != cudaSuccess) { \
            LOGERROR ("CUDA Kernel (%s) ERROR : %s\n", msg, cudaGetErrorString (_error)); \
        } \
    }

#endif


namespace CUDA {

    //  Debugging with CUDA
    // Setting 'CUDA_LAUNCH_BLOCKING' can be extremely helpful when you're trying to pinpoint 
    //  timing-sensitive, asynchronous errors like illegal memory access or race conditions.
    // Setting this variable to 1 forces each kernel to complete before the host continues, which 
    //  lets you catch errors immediately after they happen.
    void IsAsynchronousKernelLaunches() {
        const char* val = std::getenv ("CUDA_LAUNCH_BLOCKING");
        if (val) {
            LOGINFO ("CUDA_LAUNCH_BLOCKING=%s\n", val);
        } else {
            LOGINFO ("CUDA_LAUNCH_BLOCKING not set!\n");
        }
    }

    // TODO 
    //  - Make a flag or build to call this procedure.
    //

    void PrintCudaEnvironment () {
    	s32 versionRuntime, versionDriver, device, validDeviceCount;
    	cudaDeviceProp deviceProperties;

    	cudaRuntimeGetVersion   (&versionRuntime);
    	cudaDriverGetVersion    (&versionDriver);

    	// Devices with compute capability greater or equal to 1.0.
    	cudaError_t errorCode = cudaGetDeviceCount (&validDeviceCount);

    	if (errorCode == cudaErrorNoDevice) {
    		LOGERROR ("Not a single Device with compute capability greater or equal to 1.0\n");
    		return;
    	}

    	if (errorCode == cudaErrorInsufficientDriver) {
    		LOGWARN ("No driver can be loaded to determine if any devices with compute capability greater or equal to 1.0 exist!");
    		return; // This might not be necessary...
    	}
    
    	cudaGetDevice           (&device);                      // Returns the device on which the active host thread executes the device code.
    	cudaGetDeviceProperties (&deviceProperties, device);    // Gets Compute Capability and other.

    	LOGINFO (
    		"GPU-CUDA:\n\n"
    		"-----------------------------------------------------------\n"
    		"Device ID: %i, Devices Count: %i\n"
    		"Version - Runtime: %i, Driver: %i, IsIntegrated: %s\n"
    		"Name: %s, Compute Capability - MAJOR: %i, MINOR: %i\n"
    		"-----------------------------------------------------------\n\n",
    		device, validDeviceCount, 
    		versionRuntime, versionDriver, deviceProperties.integrated ? "true" : "false",
    		deviceProperties.name, deviceProperties.major, deviceProperties.minor
    	);

    	switch (deviceProperties.computeMode) {
    		case cudaComputeModeDefault:            LOGINFO ("Compute Mode: %s\n", "Default");           break;
    		case cudaComputeModeExclusive:          LOGWARN ("Compute Mode: %s\n", "Exclusive");         break;
    		case cudaComputeModeProhibited:         LOGWARN ("Compute Mode: %s\n", "Prohibited");        break;
    		case cudaComputeModeExclusiveProcess:   LOGWARN ("Compute Mode: %s\n", "ExclusiveProcess");  break;
    		default:                                LOGERROR ("Compute Mode: IMPOSSIBLE_STATEMENT\n");
    	}

    	LOGINFO (
    		"Device Properties:\n\n"
    		"-----------------------------------------------------------\n"
    		" - WarpSize:            %i\n"
    		" - MaxThreadsPerBlock:  %i\n"
    		" - MaxThreadsDim:       { x: %i, y: %i, z: %i }\n"
    		" - MaxGridSize:         { x: %i, y: %i, z: %i }\n"
    		" - MaxTexture1D:        %i\n"
    		" - MaxTexture2D:        { x: %i, y: %i }\n"
    		" - MaxTexture3D:        { x: %i, y: %i, z: %i }\n"
    		" - MaxTexture1DLayered: { x: %i, y: %i }\n"
    		" - MaxTexture2DLayered: { x: %i, y: %i, z: %i }\n"
    		" - (b) L2CacheSize:     %i\n"
    		"-----------------------------------------------------------\n\n",
    		deviceProperties.warpSize,                  deviceProperties.maxThreadsPerBlock,
    		deviceProperties.maxThreadsDim[0],          deviceProperties.maxThreadsDim[1],          deviceProperties.maxThreadsDim[2],
    		deviceProperties.maxGridSize[0],            deviceProperties.maxGridSize[1],            deviceProperties.maxGridSize[2],
    		deviceProperties.maxTexture1D,              deviceProperties.maxTexture2D[0],           deviceProperties.maxTexture2D[1],
    		deviceProperties.maxTexture3D[0],           deviceProperties.maxTexture3D[1],           deviceProperties.maxTexture3D[2],
    		deviceProperties.maxTexture1DLayered[0],    deviceProperties.maxTexture1DLayered[1],
    		deviceProperties.maxTexture2DLayered[0],    deviceProperties.maxTexture2DLayered[1],    deviceProperties.maxTexture2DLayered[2],
    		deviceProperties.l2CacheSize
    	)
    }

    // - WarpSize:            32                                        // minimum workforce group
    // - MaxThreadsPerBlock:  1024                                      // max for threads in block (MaxThreadsDim cannot exceed this)
    // - MaxThreadsDim:       { x: 1024, y: 1024, z: 64 }               // max for threads per dimension in block
    // - MaxGridSize:         { x: 2147483647, y: 65535, z: 65535 }     // max for blocks of threads

    void LogProperties (
        /* IN  */ const cudaDeviceProp& dp
    ) {
        LOGINFO (
            "Device CUDA properties:\n"
            " - WarpSize:            %i\n"
    		" - MaxThreadsPerBlock:  %i\n"
    		" - MaxThreadsDim:       { x: %i, y: %i, z: %i }\n"
    		" - MaxGridSize:         { x: %i, y: %i, z: %i }\n",
            dp.warpSize,              dp.maxThreadsPerBlock,
    	    dp.maxThreadsDim[0],      dp.maxThreadsDim[1],      dp.maxThreadsDim[2],
    	    dp.maxGridSize[0],        dp.maxGridSize[1],        dp.maxGridSize[2]
        )
    }

}
