# Accelerator Design for Convolutional Neural Network with Vertical Data Streaming  

## remark

本文介绍了加速器中数据流以及相关存储的设计，比较简略，可以参考。

## 摘要

目前硬件加速器的瓶颈在于性能和功耗，最新的CNN计算引擎需要复杂的逻辑控制来将数据送至PE。本文对数据和参数都采用流架构（streaming architecture）来降低复杂度。通过引入深入编地址模式，对ifmap的分享使得该架构能够充分利用PE。

## 2. 卷积层和数据共享

### a) 卷积层

![1](vertical%20data%20streaming.assets/1.png)

本文用到的符号如上图。步长：S≤K。有M组filter。每一个ofmap上的节点对应ifmap上的一个RoP(region of projection)，相邻的ofmap节点有重合的RoP。

### b) 卷积期间的数据共享技术

①同一组ifmap和ofmap间filter的分享，进而可以引出多ifmap的分享

②同一张ifmap的复用

## 3. 垂直数据流处理

 理想情况下，ifmap应该只流入加速器一次，随后便是ofmap的输出。但是实际情况下因为PE数量有限一次只能处理有限的ofmap，所有输入会存在很多ifmap的重复。

为了避免ifmap的重复输入，本文提出了垂直数据流技术，包含的两个方面陈述如下：

### A 深入编址模式

传统ifmap的编址如下图，是水平编址。但是由于相邻ofmap节点的RoP有重叠，这种编址会导致ifmap的重复输入。

![2](vertical%20data%20streaming.assets/2.png)

本文提出的存储ifmap的方法如下图。与同一个节点相连的ifmaps都会被连接起来存放在存储中。

![3](vertical%20data%20streaming.assets/3.png)

这种深入的编址模式也可以被运用到神经网络的参数上，比如说filter。

在这种编址模式下的计算如下图。当经过深入编址的ifmap从存储流入加速器时，它们会被所有按深入编址的ofmap同等需要。使用深入编址模式编址的filters，卷积中数据和参数的带权累加和水平编址模式下相同。得到的深入编址模式的ofmap可以直接作为下一层的ifmap。

![4](vertical%20data%20streaming.assets/4.png)

### B 垂直流架构

垂直数据流架构如下图，

![5](vertical%20data%20streaming.assets/5.png)

深入编址的数据通过位移寄存器流入和流出，参数则流入并填满每个PE的SRAM。为了最大限度上分享流入的数据，每一个PE仅计算ofmap的一个通道并且只以深入编址方式存储这个ofmap的filter。输出数据暂时存储在data memory，等待下一个层的输入。

#### a) 配置节点

架构是可以动态配置的。可以配置的节点包括每一个ifmap和ofmap的维度，卷积核的大小，是否需要padding和步长的大小。一个96位的位移寄存器会用来配置工作模式和架构，以及存储参数的位置。

#### b) 地址生成器

这个模块保证ifmap和ofmap的数据能从正确的位置读出写入。在垂直数据流模式下，ifmap可以按顺序流入并且被多个PE分享。ofmap流需要偶尔被插入写地址，因为没有足够的PE能够并行地完成所有输出通道，然而输出地址生成器在网络结构上的遵循simple prior knowledge（简单先验知识:question:）。与此相反，在水平流模式下，输入数据流不能被所有PE共享，因此PE需要额外的逻辑电路来根据RoP获取正确的数据。其他需求，比如插入padding，将会很难在水平数据流中实现，但是在垂直流模式下被大大简化了。

#### c) 对其他计算层的支持

除了卷积层，垂直数据流也可以运用于池化，激活，全连接甚至Local Region Normalization（LRN）。在地址生成器和PE内也配置了模式控制逻辑电路来切换不同的模式。为了计算不同的层，不同的层分享同一套硬件组以避免在同一个PE里实现了许多相同的算法单元。

### C 讨论

该设计的主要优点是对PE的利用率高。由于所有PE都需要同样的输入数据流来做卷积，PE的使用率达到了100%。在当前数据流出、下一级数据流入的阶段，我们会关闭PE的时钟以节省额外的动态功耗。

默认的结构中PE的数量为128，可以同时最多并行处理128个通道的ofmap。而现在领先的CNN可能拥有更多的ofmap通道，如下图所示。

![6](vertical%20data%20streaming.assets/6.png)

本设计还有一个优势，就是PE内的SRAM比较小（每个PE 6k），而不是所有PE分享一个大的PE池，这样的设计减少了同时取权重的冲突，有助于并行处理。为了控制逻辑的设计简单，当当前的ofmap输出后，参数才会流入。因此数据和参数流可以共享同一个存储，比如面积较小的单端DRAM。