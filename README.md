```
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -GNinja -DCMAKE_CUDA_COMPILER=/usr/bin/clang++
```

这条命令是在一行里调用 CMake 来配置（configure）你的项目构建，具体含义和作用如下：(最好不要让cmake工具自动去处理一些事情，工具会抽风)
- -S .
	指定「源代码」目录为当前目录（.），也就是告诉 CMake 去哪个地方找你的 CMakeLists.txt。
- -B build
	指定「构建」目录为 build，实现「out-of-source build」（源码目录外构建）。所有生成的中间文件、目标文件、以及最终的 Makefile／ Ninja 文件都放在 build/ 子目录下，不会污染源码目录。
- -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	将 CMAKE_EXPORT_COMPILE_COMMANDS 这个 cache 变量设为 ON，使 CMake 在生成构建系统时额外输出一个 compile_commands.json 文件。
- -GNinja
	指定使用 Ninja 作为底层构建系统（Generator），而不是默认的 Makefile。
- -DCMAKE_CUDA_COMPILER=/usr/bin/clang++
	手动设置 CMake 用来处理 CUDA 代码（.cu 文件）的「CUDA 编译器」。

> 这行命令会在 build/ 目录下，用 Ninja 生成一个完整的、带有 compile_commands.json 的构建系统配置，并且告诉 CMake 用 Clang 而非默认的 NVCC 去处理 CUDA 源文件。之后你只需切换到 build/ 目录，运行 ninja 就可以开始编译了。