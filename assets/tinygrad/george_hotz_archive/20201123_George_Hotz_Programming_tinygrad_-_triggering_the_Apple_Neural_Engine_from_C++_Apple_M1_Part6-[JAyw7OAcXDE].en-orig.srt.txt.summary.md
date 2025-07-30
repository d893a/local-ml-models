# George Hotz's Apple Neural Engine Reverse Engineering Stream Summary

George Hotz spent a lengthy programming stream reverse engineering how to trigger the Apple Neural Engine (ANE) from C++ on an Apple M1 device. Here's what he accomplished:

## Key Technical Progress

1. **Initial exploration**: Started by using CoreML to trigger the ANE, examining how models are compiled and loaded into the neural engine

2. **Architecture discovery**:
   - Found the "ANE Compiler Service" that compiles models into "hwx" files
   - Identified the API call chain from CoreML through Apple Neural Engine framework

3. **Low-level access**:
   - Successfully intercepted the compiler calls and understood the parameters
   - Figured out how to create CF dictionaries to communicate with the ANE services
   - Discovered the IO surface mechanism used for data transfer

4. **Direct hardware access**:
   - Learned how to create, prepare, and send requests to the ANE
   - Successfully ran a custom matrix multiplication on the neural engine
   - Verified correct outputs in float16 format

## Challenges Overcome

- Navigating Objective-C APIs and CoreFoundation
- Understanding opaque program handles and structures
- Setting up IO surfaces for input/output data
- Dealing with permission and signature verification issues

## Final Accomplishment

By the end of the stream, Hotz successfully ran a custom hwx file on the Apple Neural Engine without going through CoreML, providing a foundation for direct ANE programming. The code was committed to his TinyGrad project, potentially enabling future development of ANE acceleration for that framework.

The stream showcased reverse engineering techniques applied to a proprietary hardware accelerator, with contributions from chat participants helping overcome various technical hurdles.