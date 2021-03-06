# Eyeriss系列文章简介

eyeriss系列共有3篇文章。第一篇和第二篇发布于2016年，第三篇发布于2018年。第一篇讲述了eyeriss加速器整体的设计架构，涉及到的技术有RS数据流，RLC编码，GLB设计，NoC设计以及PE内部的设计。第二篇同期发布的文章更侧重于学术研究，比较分析了不同加速器的spatial architecture，并对其进行分类，提出了比较性能的方法，并重点介绍了eyeriss里的RS数据流。第三篇文章针对现今在移动设备上实现DNN的需求，对原来eyeriss的架构提出了修改，针对压缩的CNN提出了新的NoC结构，针对稀疏性用CSC编码代替了原来的RLC编码，对PE内部行了改进，并运用了SIMD技术提高吞吐率。此外，我还找到了一篇上交的硕士毕业论文，也是基于RISC-V用eyeriss架构制作一个加速器。它在eyeriss论文的基础上进行了进一步的推导和分析，尤其是关于kernel、ifmap、ofmap的并行，不过它是在rocket-chip上实现的，这个CPU本就有比较完备的指令扩展方式和测试平台，所用的语言也不是Verilog，并且实现的神经网络也比较简单。因此仅作为参考。

在阅读的过程中我目前遇到的最大的两个问题如下：

①PE阵列是时变的，即处理完一层后要重新配置，这个配置应该是指填入新的数据，不同层的填法也不一样，原文中只说是用比特流静态控制，但是具体怎么实现我也不是很清楚。（硕士论文里提出了一个新的自适应方法，并给出了实现的流程图。）
②eyeriss v2中PE内部也使用的是CSC编码，这是按列进行的编码，而RS结构是行不变的结构，PE内部该如何实现？还是在稀疏的情况下不用RS结构直接进行乘加，非稀疏的情况下用RS结构？