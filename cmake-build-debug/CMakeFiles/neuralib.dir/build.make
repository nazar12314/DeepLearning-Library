# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Study/AKS/semestr_project/DeepLearning-Library

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/neuralib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/neuralib.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/neuralib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/neuralib.dir/flags.make

CMakeFiles/neuralib.dir/main.cpp.o: CMakeFiles/neuralib.dir/flags.make
CMakeFiles/neuralib.dir/main.cpp.o: ../main.cpp
CMakeFiles/neuralib.dir/main.cpp.o: CMakeFiles/neuralib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/neuralib.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralib.dir/main.cpp.o -MF CMakeFiles/neuralib.dir/main.cpp.o.d -o CMakeFiles/neuralib.dir/main.cpp.o -c /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/main.cpp

CMakeFiles/neuralib.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuralib.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/main.cpp > CMakeFiles/neuralib.dir/main.cpp.i

CMakeFiles/neuralib.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuralib.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/main.cpp -o CMakeFiles/neuralib.dir/main.cpp.s

CMakeFiles/neuralib.dir/layers/Layer.cpp.o: CMakeFiles/neuralib.dir/flags.make
CMakeFiles/neuralib.dir/layers/Layer.cpp.o: ../layers/Layer.cpp
CMakeFiles/neuralib.dir/layers/Layer.cpp.o: CMakeFiles/neuralib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/neuralib.dir/layers/Layer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralib.dir/layers/Layer.cpp.o -MF CMakeFiles/neuralib.dir/layers/Layer.cpp.o.d -o CMakeFiles/neuralib.dir/layers/Layer.cpp.o -c /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/layers/Layer.cpp

CMakeFiles/neuralib.dir/layers/Layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuralib.dir/layers/Layer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/layers/Layer.cpp > CMakeFiles/neuralib.dir/layers/Layer.cpp.i

CMakeFiles/neuralib.dir/layers/Layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuralib.dir/layers/Layer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/layers/Layer.cpp -o CMakeFiles/neuralib.dir/layers/Layer.cpp.s

CMakeFiles/neuralib.dir/layers/Dense.cpp.o: CMakeFiles/neuralib.dir/flags.make
CMakeFiles/neuralib.dir/layers/Dense.cpp.o: ../layers/Dense.cpp
CMakeFiles/neuralib.dir/layers/Dense.cpp.o: CMakeFiles/neuralib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/neuralib.dir/layers/Dense.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralib.dir/layers/Dense.cpp.o -MF CMakeFiles/neuralib.dir/layers/Dense.cpp.o.d -o CMakeFiles/neuralib.dir/layers/Dense.cpp.o -c /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/layers/Dense.cpp

CMakeFiles/neuralib.dir/layers/Dense.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuralib.dir/layers/Dense.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/layers/Dense.cpp > CMakeFiles/neuralib.dir/layers/Dense.cpp.i

CMakeFiles/neuralib.dir/layers/Dense.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuralib.dir/layers/Dense.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/layers/Dense.cpp -o CMakeFiles/neuralib.dir/layers/Dense.cpp.s

CMakeFiles/neuralib.dir/utils/Initializer.cpp.o: CMakeFiles/neuralib.dir/flags.make
CMakeFiles/neuralib.dir/utils/Initializer.cpp.o: ../utils/Initializer.cpp
CMakeFiles/neuralib.dir/utils/Initializer.cpp.o: CMakeFiles/neuralib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/neuralib.dir/utils/Initializer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralib.dir/utils/Initializer.cpp.o -MF CMakeFiles/neuralib.dir/utils/Initializer.cpp.o.d -o CMakeFiles/neuralib.dir/utils/Initializer.cpp.o -c /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/utils/Initializer.cpp

CMakeFiles/neuralib.dir/utils/Initializer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuralib.dir/utils/Initializer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/utils/Initializer.cpp > CMakeFiles/neuralib.dir/utils/Initializer.cpp.i

CMakeFiles/neuralib.dir/utils/Initializer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuralib.dir/utils/Initializer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/utils/Initializer.cpp -o CMakeFiles/neuralib.dir/utils/Initializer.cpp.s

CMakeFiles/neuralib.dir/layers/Activation.cpp.o: CMakeFiles/neuralib.dir/flags.make
CMakeFiles/neuralib.dir/layers/Activation.cpp.o: ../layers/Activation.cpp
CMakeFiles/neuralib.dir/layers/Activation.cpp.o: CMakeFiles/neuralib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/neuralib.dir/layers/Activation.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralib.dir/layers/Activation.cpp.o -MF CMakeFiles/neuralib.dir/layers/Activation.cpp.o.d -o CMakeFiles/neuralib.dir/layers/Activation.cpp.o -c /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/layers/Activation.cpp

CMakeFiles/neuralib.dir/layers/Activation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuralib.dir/layers/Activation.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/layers/Activation.cpp > CMakeFiles/neuralib.dir/layers/Activation.cpp.i

CMakeFiles/neuralib.dir/layers/Activation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuralib.dir/layers/Activation.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/layers/Activation.cpp -o CMakeFiles/neuralib.dir/layers/Activation.cpp.s

CMakeFiles/neuralib.dir/utils/Optimizer.cpp.o: CMakeFiles/neuralib.dir/flags.make
CMakeFiles/neuralib.dir/utils/Optimizer.cpp.o: ../utils/Optimizer.cpp
CMakeFiles/neuralib.dir/utils/Optimizer.cpp.o: CMakeFiles/neuralib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/neuralib.dir/utils/Optimizer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralib.dir/utils/Optimizer.cpp.o -MF CMakeFiles/neuralib.dir/utils/Optimizer.cpp.o.d -o CMakeFiles/neuralib.dir/utils/Optimizer.cpp.o -c /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/utils/Optimizer.cpp

CMakeFiles/neuralib.dir/utils/Optimizer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuralib.dir/utils/Optimizer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/utils/Optimizer.cpp > CMakeFiles/neuralib.dir/utils/Optimizer.cpp.i

CMakeFiles/neuralib.dir/utils/Optimizer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuralib.dir/utils/Optimizer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/utils/Optimizer.cpp -o CMakeFiles/neuralib.dir/utils/Optimizer.cpp.s

CMakeFiles/neuralib.dir/utils/SGD.cpp.o: CMakeFiles/neuralib.dir/flags.make
CMakeFiles/neuralib.dir/utils/SGD.cpp.o: ../utils/SGD.cpp
CMakeFiles/neuralib.dir/utils/SGD.cpp.o: CMakeFiles/neuralib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/neuralib.dir/utils/SGD.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralib.dir/utils/SGD.cpp.o -MF CMakeFiles/neuralib.dir/utils/SGD.cpp.o.d -o CMakeFiles/neuralib.dir/utils/SGD.cpp.o -c /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/utils/SGD.cpp

CMakeFiles/neuralib.dir/utils/SGD.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuralib.dir/utils/SGD.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/utils/SGD.cpp > CMakeFiles/neuralib.dir/utils/SGD.cpp.i

CMakeFiles/neuralib.dir/utils/SGD.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuralib.dir/utils/SGD.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/utils/SGD.cpp -o CMakeFiles/neuralib.dir/utils/SGD.cpp.s

CMakeFiles/neuralib.dir/models/Model.cpp.o: CMakeFiles/neuralib.dir/flags.make
CMakeFiles/neuralib.dir/models/Model.cpp.o: ../models/Model.cpp
CMakeFiles/neuralib.dir/models/Model.cpp.o: CMakeFiles/neuralib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/neuralib.dir/models/Model.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralib.dir/models/Model.cpp.o -MF CMakeFiles/neuralib.dir/models/Model.cpp.o.d -o CMakeFiles/neuralib.dir/models/Model.cpp.o -c /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/models/Model.cpp

CMakeFiles/neuralib.dir/models/Model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuralib.dir/models/Model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/models/Model.cpp > CMakeFiles/neuralib.dir/models/Model.cpp.i

CMakeFiles/neuralib.dir/models/Model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuralib.dir/models/Model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/models/Model.cpp -o CMakeFiles/neuralib.dir/models/Model.cpp.s

CMakeFiles/neuralib.dir/utils/Loss.cpp.o: CMakeFiles/neuralib.dir/flags.make
CMakeFiles/neuralib.dir/utils/Loss.cpp.o: ../utils/Loss.cpp
CMakeFiles/neuralib.dir/utils/Loss.cpp.o: CMakeFiles/neuralib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/neuralib.dir/utils/Loss.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralib.dir/utils/Loss.cpp.o -MF CMakeFiles/neuralib.dir/utils/Loss.cpp.o.d -o CMakeFiles/neuralib.dir/utils/Loss.cpp.o -c /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/utils/Loss.cpp

CMakeFiles/neuralib.dir/utils/Loss.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuralib.dir/utils/Loss.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/utils/Loss.cpp > CMakeFiles/neuralib.dir/utils/Loss.cpp.i

CMakeFiles/neuralib.dir/utils/Loss.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuralib.dir/utils/Loss.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/utils/Loss.cpp -o CMakeFiles/neuralib.dir/utils/Loss.cpp.s

# Object files for target neuralib
neuralib_OBJECTS = \
"CMakeFiles/neuralib.dir/main.cpp.o" \
"CMakeFiles/neuralib.dir/layers/Layer.cpp.o" \
"CMakeFiles/neuralib.dir/layers/Dense.cpp.o" \
"CMakeFiles/neuralib.dir/utils/Initializer.cpp.o" \
"CMakeFiles/neuralib.dir/layers/Activation.cpp.o" \
"CMakeFiles/neuralib.dir/utils/Optimizer.cpp.o" \
"CMakeFiles/neuralib.dir/utils/SGD.cpp.o" \
"CMakeFiles/neuralib.dir/models/Model.cpp.o" \
"CMakeFiles/neuralib.dir/utils/Loss.cpp.o"

# External object files for target neuralib
neuralib_EXTERNAL_OBJECTS =

neuralib: CMakeFiles/neuralib.dir/main.cpp.o
neuralib: CMakeFiles/neuralib.dir/layers/Layer.cpp.o
neuralib: CMakeFiles/neuralib.dir/layers/Dense.cpp.o
neuralib: CMakeFiles/neuralib.dir/utils/Initializer.cpp.o
neuralib: CMakeFiles/neuralib.dir/layers/Activation.cpp.o
neuralib: CMakeFiles/neuralib.dir/utils/Optimizer.cpp.o
neuralib: CMakeFiles/neuralib.dir/utils/SGD.cpp.o
neuralib: CMakeFiles/neuralib.dir/models/Model.cpp.o
neuralib: CMakeFiles/neuralib.dir/utils/Loss.cpp.o
neuralib: CMakeFiles/neuralib.dir/build.make
neuralib: /usr/lib/x86_64-linux-gnu/libboost_program_options.so.1.74.0
neuralib: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
neuralib: CMakeFiles/neuralib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable neuralib"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/neuralib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/neuralib.dir/build: neuralib
.PHONY : CMakeFiles/neuralib.dir/build

CMakeFiles/neuralib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/neuralib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/neuralib.dir/clean

CMakeFiles/neuralib.dir/depend:
	cd /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Study/AKS/semestr_project/DeepLearning-Library /mnt/c/Study/AKS/semestr_project/DeepLearning-Library /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug /mnt/c/Study/AKS/semestr_project/DeepLearning-Library/cmake-build-debug/CMakeFiles/neuralib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/neuralib.dir/depend
