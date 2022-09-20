# 基础知识

# 方法论

# 常用数据集

 FUNSD:199张文档图片
 
 SROIE:1000张小票图片
 
 EPHOIE:1494张中文试卷

# 关键信息提取论文

## 1.[ACL2022]FormNet:表单文档信息提取中超越序列建模的结构编码方法

### 一、研究背景
表格类文档的理解是一个越来越受到广泛关注的研究课题，它自动提取和组织有价值的文本数据，能实际应用于如抽取各类表单、广告和收据的信息过程中。理解自然语言表示的典型文档、文章或互联网中的内容已经得到广泛研究。然而，类似于表格的文档往往具有更复杂布局，包含结构化对象如表格和列。因此，与自然语言文档相比，表单类文档因其结构特征是一项具有独特挑战的课题，目前还没有进行充分探索。
随着序列化建模在自然语言理解任务中取得成功，这种方法也被广泛应用于文档理解领域。首先将表单文档序列化，然后将最先进的序列模型应用于其中，[1]采用Seq2seq with RNN，[2]采用Transformer。同时也有许多工作[3][4]关注文档的版面结构信息。然而这些工作在处理表单这种高度结构化的文档时依然无法取得足够好的效果。 

### 二、方法原理简述
![image](https://user-images.githubusercontent.com/29652509/177334119-dc4edf51-98eb-4d87-93f6-a1fb660027b6.png)

上图是这篇文章提出的FormNet的整体结构，网络对于传统方法的改进有两点，一是使用图卷积网络GCN建模文档结构信息，二是改进Transformer，使用ETC[5]（采用全局-局部注意力机制的Transformer），在处理长序列建模任务时有效降低了模型的计算复杂度，并加入Rich Attention，计算注意力分数时加上相关惩罚偏置，使注意力更好地关注到文档中结构相关部分。
图卷积网络GCN部分首先使用β骨架图生成方法（β=1）对Token化后文档生成连通图，如图4所示，这种灵活且有效的连接有利于后续对文档结构信息的获取。接着生成节点和边后送入12层图卷积网络GCN，得到节点特征送入后续网络进一步建模。
ETC部分在Transformer的基础上改进注意力计算，采用全局-局部注意力机制，计算注意力时对于当前Token仅有序列中在距离r半径内的其他Token会计算注意力，较远Token对当前Token完全不可见。同时本文为注意力分数计算设计了一个引入与Token顺序和距离关系相关惩罚偏置的方法Rich Attention，进一步限制了注意力的计算，使网络更关注Token的序列关系，也即更好地获取了文档结构信息，Rich Attention结构如图5所示，对注意力分数的限制可视化如图6所示。
网络采用常用的预训练任务Masked Language Model（MLM），随机掩码恢复Token任务，进行预训练。使用相对于主流预训练模型（11M）显著较少的预训练数据（9M）即可在微调后在下游任务中达到SOTA，如图7所示。

### 三、实验结果
![image](https://user-images.githubusercontent.com/29652509/177334721-d2a734dd-7f66-437c-95ab-be109619f9b6.png)
![image](https://user-images.githubusercontent.com/29652509/177334734-284f12cc-d4d8-4bdc-b479-53439c4d7f49.png)


本文在三个表单类信息抽取数据集上测试了方法的效果，如上图表所示。图4图6分别可视化展示了图卷积网络和Rich Attention的效果，在表2中进一步消融实验证明了二者的作用。图7对比了各模型参数和在CORD数据集上的效果，FormNet实现了模型参数量和模型效果的最优平衡。表1还分析对比了各预训练方法的预训练数据大小，证明了本文提出方法在少量预训练数据下的优越性。

## 2.[ICDAR2021 BestPaper] ViBERTgrid：一种用于文档关键信息提取的联合训练多模态二维文档表示

### 一、研究背景

最近基于Grid的文档表示，如BERTGrid，允许同时编码文档的文本和布局信息在一个二维特征映射图，使得一些SOTA的图像分割/目标检测模型可以直接用于提取文档关键信息。然而，目前这些方法的性能还没有达到与基于序列和基于图的方法（如LayoutLM[1]，PICK[2]，TRIE[3]以及VIES[4]）相当的水平。图1比较了不同文档表示类型的先进方法，可以看出目前最先进的基于序列和基于图的方法采用了一些重要的技术来提高精度，包括融合不同模态的信息，联合训练不同模态特征，引入NLP领域最新的BERT类预训练语言模型以及借助基于文档图像的大规模预训练等。然而与之相比，之前最先进的grid文档表示方法，如BERTgrid[5]和VisualWordGrid[6]，却在这些方面有着极大的缺失。基于这个观察，本文提出一种新的grid类型文档表征方法ViBERTgrid，并以此研究grid的文档表示是否也能在文档理解场景下得到最好的性能。为了验证本文提出的文档表征方法的有效性，作者在关键信息提取任务（KIE）上进行了实验。关键信息提取的任务是从如发票、采购订单、收据等文档中提取许多预定义的关键字段的值。

![image](https://user-images.githubusercontent.com/29652509/177335610-797ab2a3-0d42-462c-a528-da7f7a747eeb.png)

图1. 不同文档表示类型的先进方法之间的比较。

### 二、方法简述
如图2所示，本文提出的方法由三个关键组件组成：1)用于生成ViBERTGrid特征图的多模态主干网络；2)辅助语义分割头，用于逐像素字段类型分类；3)词级字段类型分类头，用于预测每个单词的字段类型。

![image](https://user-images.githubusercontent.com/29652509/177335718-10bedcc7-f364-4ba8-945a-29baaf46156f.png)

图2. 基于ViBERTGrid文档表示构建的关键信息提取方法框架图

 #### 2.1. ViBERTgrid生成
遵循BERTGrid[5]的方法，首先构建一个二维单词grid图，其中的单词嵌入由预先训练的BERT生成，得到的Grid图可以表示为如下：
![image](https://user-images.githubusercontent.com/29652509/177336778-2cd86d5e-b11b-4f5b-8353-7a356b5fa183.png)

文档图中含有单词的位置用该单词的词嵌入表示，其他位置则用零向量表示。接下来，将生成的BERTGrid 图拼接到CNN的中间层，从而构建一个新的多模态主干网络，具体结构如图3所示。为了节省计算量，CNN采用了构建在ResNet18-D网络之上的轻量级ResNet18-FPN网络，而新增的BERTGrid 图拼接到Conv3_1卷积层之后的特征图中。

#### 2.2. 词级别字段类型分类头
通过上面的ViBERTGrid主干网络之后可以得到文档图像的特征图，对于每个单词，用ROIAlign[8]来获取单词级别的输出特征。然后用两层3 x 3卷积层将其映射成一个小的7 × 7 × 256特征图，并用全连接层转化为一个1024维的向量。最后，将该特征向量与BERT输出的单词嵌入向量拼接起来，输入后面的字段类型分类头。字段类型分类头包含两个分类器：第一个是一个二分类器，来判断某个单词是否属于指定的某个字段类型；第二个分类器则对属于指定某个字段类型的单词作进一步的分类，判断是具体的哪个字段类型。两分类器均使用交叉熵损失进行监督训练，最后进行损失叠加，可表示为如下（其中k表示字段类型数）：
![image](https://user-images.githubusercontent.com/29652509/177336907-7b4da867-65df-44b3-95c9-6b728a5facda.png)

#### 2.3. 辅助语义分割头
在训练这个端到端网络时，本文发现增加一个额外的像素级分割损失可以使网络更快更稳定地收敛。类似于词级字段类型分类头，这个辅助语义分割头也包含两个分类器：第一个是判断像素点属于指定字段某个类型、无关文本类型或背景；第二个分类器则继续将像素点进一步分类为具体的字段类别。同样，损失可以表示为：
![image](https://user-images.githubusercontent.com/29652509/177337130-984d31c3-3638-498d-94c8-95998e3a1c18.png)

#### 2.4. 联合训练策略
联合训练BERT模型和CNN模型并不容易，因为这两种模型是通过不同的优化器和超参数进行微调的。预先训练的BERT模型通常使用Adam优化器进行微调，学习率非常小，而CNN模型使用标准SGD优化器进行微调，学习率相对较大。因此，本文分别用AdamW[9]优化器和SGD优化器对预训练的BERT模型和CNN模型进行训练。这两个优化器的超参数设置也是独立的。

### 三、实验设置与结果
#### 3.1. 与已有方法的比较
本文在INVOICE和SROIE数据集上，和目前主流的信息提取方法作比较，得到的结果如Table 2与Table 3所示。可以发现，本文提出的方法可以取得相当甚至更好的性能
![image](https://user-images.githubusercontent.com/29652509/177337189-45a9bda6-0fee-4abd-a528-cbdfc604038b.png)
![image](https://user-images.githubusercontent.com/29652509/177337211-173a6a6d-7a2a-443c-9a8f-5b339fb50d6e.png)

#### 3.2. 消融实验
本文设置消融实验分别比较了联合训练、不同模态特征的使用、对不同预训练语言模型加入CNN模型、BERTgrid图加在CNN特征图的不同阶段、在CNN中间层或特征输出层分别加入BERTgrid图和词嵌入特征带来的影响，得到的结果如Table 4-8。
![image](https://user-images.githubusercontent.com/29652509/177337453-f0211d48-2eeb-48db-aec9-433be02f7fcd.png)

从上述的消融实验结果中可以得到结论：本文提出的使用不同的优化器、不同的学习率来针对BERT与CNN的联合优化方法具有最好的性能；两个模态特征同时使用，并联合优化可以实现最优性能；CNN模型的使用对提升方法性能作用显著；将BERTGrid图添加到CNN的C3阶段特征图能有较好的性能；在不同的模型层数条件下，同时在CNN中间层和特征的输出层加入BERTgrid图和词嵌入特征都能有最好的性能。

## 3.[IJCAI 2021] MatchVIE: 一个基于命名实体匹配相关性的视觉信息抽取学习框架

### 一、研究背景
视觉信息提取 (VIE) 旨在从文档图像（发票、购买收据、身份证等）而不是纯文本中提取关键信息。VIE 任务的特殊性为信息抽取任务带来了一些额外的困难。首先，文档通常具有不同的布局，即使对于相同类型的文档（例如，来自不同供应商的发票），布局也有很大差异。此外，视觉富文档图像中可能包含多个相似但属性不相同的文本（例如，签发日期和到期日期）。因此，如何有效地利用语义特征和视觉特征提高模型对于信息抽取的鲁棒性成为近期研究的重点。因此，一些方法尝试将语义特征与文本的位置，图像等特征融合以进一步提高性能。以前的大多数方法都局限于序列标记或直接分类方法，这就需要模型在面对众多语义类别或一些含糊不清的文本时需要仔细地为每个实体分配相应的标签。 

![image](https://user-images.githubusercontent.com/29652509/177338744-cb5b7f68-15cb-441e-bc03-4cbdd25a53cc.png)

Figure 1 The entity categories of the blue text segments (value) can be identified according to the semantics of connected red text segments (key). The entity categories of the yellow text segments can be identified according themselves semantics.

实际上，如文献[1]验证了视觉丰富的文档中实体之间的布局信息对判断实体之间的属性起着至关重要的作用。如Fig.1所示，蓝色文本段（值）的实体类别可以根据红色文本段（键）的语义确定。相比于序列标注或是纯分类的方法，本文发现研究键和值之间的相关性可能是 VIE 的另一种有效解决方案，基于以下观察和考虑：（1）文档图像中的文本通常以键值对的形式出现。如果能找到该值对应的键，自然可以根据键的语义来确定该值的属性。(2) 一个文档图像中可能有多个相似的文本（例如，注册日期和金额数字等），这些实体对应的键可以帮助模型更好地区分它们。(3) 考虑键和值之间的相关性可以显著地简化模型的学习过程并绕过对相似语义的序列标注。(4) 对于一些独立的实体，也就是不构成键值对的实体，它们很容易通过其自身的语义确定它的实体属性。这也是这些文本可以单独出现在文档图像中的原因。 

![image](https://user-images.githubusercontent.com/29652509/177338807-c1508cd1-25a5-4125-b92d-b2b75459bf2f.png)

Figure 2  Overall framework of MatchVIE

### 二、方法简述
为了解决上述提到的问题，本文提出了一种称为MatchVIE 框架。如Fig. 2 所示，它分别由一个多模态特征提取主干网和两个特定分支（相关性评估分支， 实体识别分支）组成。多模态特征提取主干网结合位置、图像和语义等嵌入信息。随后，相关性评估分支基于图模块对文档的整体布局信息进行建模并获得键值匹配概率。同时，为了解决独立文本段的序列标签识别问题，作者设计了一个实体识别分支。
本文在多模态特征提取主干网中尝试提取丰富的模态特征，例如语义，位置和视觉特征。在视觉特征部分，与一些方法从预先裁剪出的文本块图像中提取视觉特征不同，本文方法先将整图送入ResNet主干网，随后采用ROI-Align提取文本段的坐标区域对应的特征图。 

![image](https://user-images.githubusercontent.com/29652509/177338915-fc97fd0a-4809-4068-b252-da821e0481e1.png)

这样的好处在于保留了全局的特征信息。随后，针对提取的多模态特征，引入自注意力机制对多模态特征做特征融合。  
 
为了表示实体之间的相关性，文本段之间的位置关系是一个重要的推理线索。因此本文构建了一个文档图结构，意味着，每个文本段作为一个单个节点，初始的节点特征提取每个文本段中所有字符的多模态特征得到。而文本段之间的相对位置关系反映的是图网络中边的特征。以往的方法，在量化节点之间的相对位置关系的时候通常采用如下的表征方法。 

![image](https://user-images.githubusercontent.com/29652509/177338961-6a8e3ffe-ad2c-4716-bb91-0e86e8099aab.png)

然而，本文发现由于不同文本段之间的距离和形状的变化多样，这就会导致编码的数值非常不稳定。为了解决这个问题，本文采用一种简单而有效的操作，即 Num2Vec 来处理每个值。如Fig. 2 所示，对于每一个数值，本文采用一个固定长度的8位数组来保存每个数字。数组的前半部分对应于整数部分，其余部分对应于小数部分。然后，将这些数字按因子 0.1 进行缩放。这样就保证了编码值被限制在[-0.9,+0.9]的范围内，可以有效地减小数据的波动范围。
在 GNN 中特征更新的过程中，遵循文献[2]采用的方法，其定义了一个三元组 (节点-边-节点) 的形式用于特征更新。三元组特征由一个可学习的权重进行线性变换，生成一个中间特征向量，进行特征更新。
假设有N个节点，则所有文本段的相关性匹配矩阵可以枚举为N×N。而任意两个文本段之间的匹配关系可以建模成二值分类任务。如果两个文本段之间构成键-值关系，则用1表示。此外，由于N×N矩阵中正样本的数量是比较稀少的。为了处理这个问题，本文对相关性评估的损失函数采用Focal Loss[3]： 

![image](https://user-images.githubusercontent.com/29652509/177339018-d4eeabe1-cdaf-4f9e-91e2-951781c302ce.png)

同时为了解决独立实体的属性区分问题，本文还设计了一个实体识别分支。这个分支将字符级别的多模态编码特征采用经典的序列标注方法，结合CRF来约束实体的句法关系。
模型在训练的时候，对相关性评估分支和实体识别分支的损失函数进行联合优化。在前向推理的时候，优先采用相关性评估分支的输出，对于一些不构成匹配关系的实体，视其为独立类别，采用实体识别分支分支的输出作为最终结果。此外，对于存在实体关联性的键值对，其实体属性的确定方式有两种方式，第一种方式是穷举每个实体属性中键所有可能存在的文本形式，将其构建为一个查找表。第二种方式是学习每个实体属性中键的文本的语义属性，将特征最接近的类别作为该键值对的类别。 

![image](https://user-images.githubusercontent.com/29652509/177339060-bbbccc31-f92a-45fd-8546-f0b737e7d5cb.png)

### 三、实验结果

![image](https://user-images.githubusercontent.com/29652509/177339107-49c9c8da-450f-4c29-8a34-804624aee43b.png)

Table 1  Ablation study (F1-score) of the proposed model on the three datasets. 

如Table 1所示，作者设置了消融实验分析了MatchVIE中各个组件的影响，包括Focal Loss、K-V匹配、Num2Vec。模型逐渐去掉这三个模块，分别测试了模型精度的变化情况。可以发现，不采用Focal Loss损失函数，相关性评估分支无法有效克服匹配矩阵非常稀疏的问题。可以看出，相关性评估分支（K-V 匹配）可以大幅度提高准确率。为了进一步验证相关性评估分支的有效性，本文分别给出了是否结合相关性评估分支模型的预测结果。当相关性评估分支模型去除以后，模型变成了结合多模态信息的纯序列标注方法。可以看到基于序列标注的方法由于预先需要按照从左到右，从上到下的顺序预先组织文本的顺序，其过度依赖于预先定义好的先后顺序，并不能很好的处理这种多模态特征比较接近的数字语义类别。而本文提出的方法通过引入命名实体之间的相关性来有效区分这些类别。 
此外，在使用 Num2Vec后，模型可以获得更稳定的结果，并具有额外的精度改进。此外，本文收集训练损失函数并绘制了损失曲线。从Fig. 4 可以看出，Num2Vec 可以帮助模型更平滑地收敛。 

![image](https://user-images.githubusercontent.com/29652509/177339160-698394e6-701f-40d3-89aa-ce38ebd978c0.png)

 Figure 4  The yellow polyline indicates loss without Num2Vec and the blue polyline indicates the loss with Num2Vec. It can be seen from that using Num2Vec can make training converge smoothly.

本文还尝试了两种方式来将一个键映射到某个类别， 两种方式的对比结果如Table 2 所示。一个是穷举每个类别所有可能键的文本值（例如，类别“Total”的键的查找表可以是“Total Sales”、“Total”、“Total Amount”、“Total (RM)”） ，另一个是基于特征的相似性。 

![image](https://user-images.githubusercontent.com/29652509/177339221-9b0c5aad-ec90-4f00-948f-3ca8108993ac.png)

作者将本文提出的MatchVIE方法和之前的主流方法进行对比，结果如Table 3所示。可以看到， MatchVIE方法绕过对各种语义的识别而只关注实体之间的强相关性，从而取得了不错的结果，特别针对一些实体本身不具备语义信息或是语义信息存在歧义性的实体类别，例如数字，日期等。通过实体相关性的引入，这些类别的属性得到了很好地区分。

![image](https://user-images.githubusercontent.com/29652509/177339289-659fdcec-c131-4efe-80df-4120f12efc46.png)

Table 3  Experiment results on EPHOIE datasets. Standard F1-score (F1) are employed as evaluation metrics. NO: Number, STU:Student, EXAM: Examination. 

## 4.[IJCAI 2021]TCPN: 一个基于序列的弱监督视觉信息抽取学习框架

### 一、研究背景

随着信息交互的日趋频繁，文档智能处理技术已经引起了社会广泛的关注。作为其中的重要组成部分，视觉信息抽取方法已被集成到诸多日常实际应用中。现有的视觉信息提取方法通常会根据自然阅读顺序将文本块（文本外包围框和对应的文本字符串）组织成一维的纯文本序列，并利用一些有效的编码器结构从多个模态（文本，版面，视觉等）中为每个输入字符提取出最有效的特征表示。字符级实体类别监督信息会用来训练一个如BiLSTM-CRF[1]的序列标记模型。 

![image](https://user-images.githubusercontent.com/29652509/177358408-f3d9c5f3-0c7e-4865-83d8-035445e3e2c4.png)

图1 Illustration of the annotations required by the traditional method and our weakly-supervised framework

然而，字符级别的实体类别标签往往会花费大量的标注成本，并且可能会面临标注歧义问题。给定如图1(a)所示的文档图像，传统的标注方案是对每个文本段的外包围框和对应的文本字符串进行标注，并进一步指出每个字符或文本段属于哪个实体类别，如图1(b)所示。这时，我们会需要一个启发式的标签匹配准则来达到训练之前提到的序列标记模型的目的，而其核心思想就是将OCR系统的检测识别结果与给定的真实标签进行匹配，然后为 OCR 结果中的每个字符或文本段分配标签。然而，这个过程可能会面临两个方面的问题：首先，文字检测和识别误差会给匹配的过程带来困难，尤其是对于关键信息序列部分而言；其次，重复的内容会带来标签的匹配歧义问题。如图1(a)和(b)所示，三段内容相同的文本“8.20”都可以被视为“总金额”这个实体类别的答案。而在大多数情况下，很难建立一个统一的标注规范来决定此时应该将重复文本中的哪一个视为真实标签。 


图2 The overall framework of TCPN. (a) The input OCR results (the recognition result is ignored for visual clarity). (b) The corresponding TextLattice I. (c) The light-weight modified ResNet encoder combined with the U-Net architecture. (d) The result of feature encoding, which has the same size as I. The red bounding box shows a wrongly recognized character ‘M’. In Tag Mode, it can be classified in a single forward pass; while in Copy or Predict Mode, it can be corrected as‘N’.

### 二、方法简述

为了解决上述提到的问题，本文提出了一种统一的端到端弱监督学习框架TCPN（Tag, Copy or Predict Network），如图2所示。该框架主要由三个部分组成：1）一种高效的二维文档表示形式TextLattice，和对应的轻量化编码器结构；2）一个包含两种前向推理模式的解码器结构——具备OCR纠错功能的TCPN-CP模式和推理速度较快的TCPN-T模式；3）一种可以直接使用关键信息序列指导训练过程的弱监督学习范式。该框架可以带来诸多好处，例如：1）大大节省了标注成本，如图1(c)所示。同时避免了OCR结果与真实标签的人工匹配过程；2）该方法可以自动学习 OCR 结果和真实标签之间的对齐关系来解决标签歧义问题；3）该方法具备OCR纠错的能力，一定程度上缓解了OCR误差对后续任务的影响。以下将分别就每个方面进行介绍。
为了进行更加高效的特征提取过程，本文提出了一种2维文档表示形式TextLattice和对应的编码器结构，如图1(b)(c)(d)所示。给定一份OCR检测识别结果，该方法执行如下过程：首先对检测框的y坐标归一化处理，将检测框按照从左上到右下的顺序排列并划分为多行；接着，将文本段级别的框切分成字符级别的框，并微调x坐标避免重叠；最后，初始化一个全0矩阵，并在相应位置填入字符级映射向量。详细过程以及可视化结果可参照论文中附录部分。
本文使用ResNet[2]结合U-Net[3]结构作为特征编码器部分。通过这种方式，网络可以将不同感受野下的局部和全局特征进行自适应的融合。同时，为了更好的感知整体版面信息，本文借鉴CoordConv[4]的思想，额外将x和y方向上的相对位置坐标信息拼接到TextLattice中，整个过程如下所示： 

![image](https://user-images.githubusercontent.com/29652509/177358478-7a6e5d12-781e-4de8-bcda-fe86fb687eac.png)

由于U-Net的输出结果与输入I大小相同，因此文中采用了残差结构进行特征加和。最后，每个字符所对应的特征向量会根据之前得到的相应位置坐标被取出，而剩下的像素点位置特征会被直接丢弃。
在弱监督训练部分，文中首先提出了实体类别映射向量的概念，来控制解码器输出的信息类别，而其本身也是从一个预定义好的可训练的查找表矩阵中随用随取。给定该向量，解码器可以在每个时间步考虑当前需要生成的实体类别，并迭代预测得到信息序列。这种方式避免了为每个实体类别独立设计解码器，同时也缓解了单一类别语料不足的影响。当生成信息序列时，本文希望模型能够在“复制输入中的字符”和“直接预测字符”两个操作间切换。前者是为了让模型具备保留输入中稀有词的能力，而后者使模型拥有了进行OCR纠错的能力。本文中模型会迭代生成隐状态向量以进行预测： 

![image](https://user-images.githubusercontent.com/29652509/177358503-072bac0a-016a-43e3-843e-0a464cf94efc.png)

接着，模型会计算字典中字符的概率分布，以及一个复制概率分数来决定每个时间步应该执行的最优操作是复制还是预测： 

![image](https://user-images.githubusercontent.com/29652509/177358520-86321dec-23fc-40a4-8001-0ac22e0a2454.png)

通过这种方式，模型也具备了生成OOV（Out-of-vocabulary）字符的能力。
到目前为止，本文的方法还只是一个纯粹的使用序列信息监督进行训练的序列生成模型。然而，本文的核心思想是：既然给定了类别c的映射向量，当模型在某个时间步认为需要从输入字符中复制字符k时，那么k所对应的特征向量应该可以被一个分类器分类为实体类别c。换句话说，解码器可以先学会如何进行字符对齐，然后利用对齐的字符训练一个线性分类器。该思想使该框架具备了仅使用序列级信息就可以监督序列标记模型训练的能力。使用线性分类器预测实体类别的过程如下所示： 

![image](https://user-images.githubusercontent.com/29652509/177358587-f7590ebd-45f7-43e0-9406-cbc375e76271.png)

值得注意的是，上述公式只训练了属于关键信息部分的字符。因此，本文又额外加入了一个辅助损失函数来压制负样本被预测为正样本的数量： 

![image](https://user-images.githubusercontent.com/29652509/177358618-8c4629bd-c38b-4b40-be5b-314515e8d053.png)

该损失函数的主要目的是限制输入中被分类为实体类别c的字符个数不能超过类别c对应的关键信息序列的真实长度。总的来说，训练过程中整体的损失函数是上述提到的几种损失函数的加权和： 

![image](https://user-images.githubusercontent.com/29652509/177358646-278ac3a4-422f-4637-8fe4-c3400a5ed483.png)


模型在执行前向过程时，不同实体类别可以根据自身的特性采用不同的解码模式。当某个类别的语义相关性较强时，可以使用文中公式（3）-（9）的TCPN-CP模式执行OCR
纠错的过程；当某个类别的语义相关性较弱或者OCR错误极少时，可以使用文中公式（14）的TCPN-T模式执行序列标记的过程。

### 三、实验结果

Table 1  Performance and speed comparison on EPHOIE dataset between different encoding architectures. FPS is tested on a GeForce GTX 1080 Ti. 

![image](https://user-images.githubusercontent.com/29652509/177358787-57703035-edb8-46f4-a969-9d597aa2d920.png)

作者首先将本文提出的文档编码方式和之前的主流方法进行对比，结果如Table 1所示。可以发现，TextLattice在保持较高计算效率的条件下同时具有最好的效果。该方法相比GAT in [5]和BERT-like[6]中的位置编码方式具有对位置特征更加直观的感知，同时相比Chargrid[7]具有更高的信息集中程度。

Table 2 Effects of different components in the proposed encoding architecture on EPHOIE dataset. 

![image](https://user-images.githubusercontent.com/29652509/177358785-c3419dc5-8e3d-4011-8266-19555de27c44.png)


作者同样对编码器中的不同结构进行了消融实验，结果如Table 2所示。不难发现，不管是CoordConv、U-Net结构还是残差连接，都对模型的表现有正面影响。

![image](https://user-images.githubusercontent.com/29652509/177358786-bc4c6804-8baa-4c4e-9987-ccd3dad5de82.png)

Table 3 Performance comparison on (a) EPHOIE and (b) SROIE under ground truth setting. ‘Token-level Fully-Supervised’ indicates the token-level category annotation, and ‘Sequence-level Weakly Supervised’ means the key information sequence annotation. 

为了验证方法的有效性，作者将本文提出的TCPN和以往的主流方法在SROIE[8]和EPHOIE[9]数据集上进行了对比。
在Ground Truth Setting设置下的实验结果如Table 3所示。可以发现，在使用字符级全监督时，该方法在两个数据集上都表现较好，这充分证明了TextLattice的有效性；同时，在序列级弱监督时，该方法也可以取得令人满意的结果，这充分证明了本文提出的弱监督学习方法的有效性。作者指出，和SROIE相比，EPHOIE数据集中往往包含的文本内容更少，且字符的种类数更多，这降低了模型学习字符对齐关系的难度。相反，SROIE中的票据文档往往包含大量的字符，且同一字符往往重复出现多次，这也导致了全监督和弱监督学习方式下效果的差距更大。

![image](https://user-images.githubusercontent.com/29652509/177358839-7a83bf29-0e74-4fc2-b478-35d1519e241b.png)

Table 4 Performance comparison on (a) EPHOIE and (b) SROIE under end-to-end setting. ‘Rule-based Matching’ indicates acquiring token-level label through traditional rule-based matching, and ‘Automatic Alignment’ means automatically learning the alignment using the key information sequences. 

本文也在End-to-End Setting设置下进行了实验，其结果如Table 4所示。由于OCR结果中不可避免的会存在误差，所以此处本方法的实验都是在弱监督的条件下进行，这也意味着模型会自动学习字符对齐关系。可以发现，不管是在TCPN-T模式还是TCPN-CP模式下，本文的方法都有更好的表现。作者也指出，选择何种模式的一个重要依据是语义的丰富程度和语料的充足程度。在SROIE数据集上，TCPN-CP模式的结果优于TCPN-T，这正是得益于模型的OCR纠错能力；而在EPHOIE数据集上，TCPN-T模式大幅优于TCPN-CP，其主要原因就在于中文字符种类繁多，相对应的导致语料较匮乏。

![image](https://user-images.githubusercontent.com/29652509/177358851-1094948c-f052-449c-b3f1-7a74e9fca9cd.png)

Table 5 Performance comparison on Business License Dataset under end-to-end setting.  

为了验证本文方法在实际应用场景中的效果，作者也在一个从真实用户处收集的内部营业执照数据集上进行了实验。该数据集包含1863张训练图片和468张测试图片，设置了13种待提取的实体类别。由于这些图片都是由移动设备捕捉，会存在大量模糊、扭曲、曝光、折叠等条件的负面影响，其OCR结果中也必然存在误差。实验结果如Table 5所示。本文提出的端到端弱监督学习框架显著超越传统的基于规则匹配的方法，同时也节省了标注成本。TCPN-CP模式下所学习到的隐式语义关联性让该模型具备了良好的OCR纠错能力。可视化结果见论文附录部分。

### 四、可视化结果

![image](https://user-images.githubusercontent.com/29652509/177356969-764772d9-1cf5-4c04-a76d-253e8feb3494.png)

![image](https://user-images.githubusercontent.com/29652509/177356998-59d1f7de-2d28-4cc5-81ad-b7fb505ef52f.png)

![image](https://user-images.githubusercontent.com/29652509/177357010-e1667a91-d6c9-4cc3-86df-90c4fde080bb.png)

### 五、总结及讨论

本文提出了一个统一的弱监督视觉信息抽取学习框架TCPN，它包含1）一种高效的文档表示形式TextLattice和对应的轻量化编码器结构；2）一个包含两种前向推理模式的解码器结构；3）一种可以直接使用关键信息序列指导训练过程的弱监督学习范式。该方法着眼于传统视觉信息抽取框架中存在的对细粒度人工标注过度依赖、标签匹配面临歧义、模型表现易受OCR 误差影响等问题，进行了充分合理的改进，在多个数据集上都取得了最佳的性能，这也充分验证了该方法的有效性。总结来说，本文方法的主要优点有：
- 大大节省了标注成本，如图1(c)所示;
- 避免了OCR结果与真实标签的人工匹配处理过程；
- 该方法可自动学习 OCR 结果和真实标签之间的对齐关系来解决标签歧义问题；
- 该方法具备一定的OCR纠错的能力，一定程度上缓解了OCR误差对后续IE任务的影响。

## 5.[AAAI 2021]面向真实场景的视觉文档信息抽取：新数据集和新解决方案

Towards Robust Visual Information Extraction in Real World: New Dataset and Novel Solution

### 一、研究背景

近年来，可视信息抽取技术受到越发广泛的关注。其在如文档理解、信息检索和智能教育等诸多时下热门的任务场景中得到广泛应用。现有的可视信息抽取方法主要分为两个独立的阶段：1）文本检测与识别；2）信息抽取。前者用来得到图片中所包含的全部文本的位置与内容，而后者在前者提供的结果上，进一步提取出特定类别的关键信息。然而，现存方法的局限性主要在于：1）尽管文本检测与识别模型已经学习到有效的特征表示，但在信息抽取部分，这些特征被直接丢弃，而又从OCR结果中重新生成。这导致了计算资源的浪费，并且被丢弃的特征可能比重新学习到的更有效；2）模块间的独立性导致他们的训练过程没有交互，这一方面限制了信息抽取模块所能获得的有用信息量，另一方面也使得文本检测与识别模块无法根据最终目标进行自适应的优化。
随着深度学习方法的蓬勃发展，针对某一特定领域所组建的全面且公开的数据集基准是激励未来工作的重要前提。在VIE领域，SROIE[1]是时下应用最广泛的数据集基准，它同时囊括了OCR与VIE任务，且面向印刷体英文的扫描票据场景。然而，它无法充分满足现实应用中对于复杂版面、手写体文字以及中文文档的需求。 

### 二、数据集（EPHOIE）简述

本文提出了一个称为EPHOIE（Examination Paper Head dataset for OCR and Information Extraction）的新数据集基准，是第一个同时兼顾OCR与VIE任务的中文数据集，旨在进一步推动该领域的发展。它同时囊括手写体和印刷体字符，共包含1494张图像，且被划分为1183张图片的训练集和311张图片的测试集。数据集中所有的图片都是从真实的考试试卷中收集扫描得到的不同学校、不同板式的试卷头信息。一些图片如图1所示。

![image](https://user-images.githubusercontent.com/29652509/177359233-d0e0022d-4d30-446a-8361-fb2fbf1eab81.png)

图1  EPHOIE数据集中的一些图像展示 

表1  EPHOIE与SROIE数据集的对比

![image](https://user-images.githubusercontent.com/29652509/177359299-09204634-3e2c-41a8-b4d2-b57184e6796c.png)

文中将EPHOIE数据集与目前应用最广泛的SROIE数据集进行了比较，结果如表1所示。

![image](https://user-images.githubusercontent.com/29652509/177359321-ab21f266-a37a-4aa6-aa4b-b3ab43bbfb1b.png)

图2  EPHOIE数据集标注格式

图2展示了EPHOIE数据集的详细标注格式。由于该数据集中同时存在水平和任意四边形文本框，所以使用四个顶点表示。同时，文本内容以及对应的实体类别和键值对属性也进行了标注。‘Entity’字段中的数字字符串表示内容对应的实体，这种字符级细粒度的标注是为了应对单个文本段中存在多种实体的情况。

### 三、方法（VIES）简述

![image](https://user-images.githubusercontent.com/29652509/177359409-9d1d280b-e8dd-4933-a9c0-c15aa8c91035.png)

图3  本文方法整体框架图

本文提出的方法的总体框架如图3所示。它由一个共享主干网络和三个特定的子任务分支——文本检测、识别和信息抽取分支组成。给定一张文档图像，文本检测与识别分支不仅负责定位并识别图中包含的所有文本，同时还通过文中提出的视觉与语义协作机制（Vision And Semantics Coordination Mechanism，VCM and SCM）为后续网络提供丰富的视觉和语义特征。信息提取分支中提出的自适应特征融合模块（Adaptive Feature Fusion Module，AFFM）收集多模态的特征表示，并利用这些特征自适应地生成不同细粒度的融合信息。接下来将对各个子分支进行详细介绍。

##### 1）文本检测分支

给定输入图像，本文首先使用共享主干网络提取高级特征表示X。然后，检测分支采用类似Mask R-CNN[2]的结构将X作为输入，并输出检测框B、置信度C以及为任意四边形框准备的掩码M：

![image](https://user-images.githubusercontent.com/29652509/177359497-e9abf427-f123-4f33-a90e-a9ae85dbe69c.png)

此处，该工作提出了视觉协作机制（VCM），以此将丰富的视觉特征从检测分支送至信息抽取分支，同时也相对的提供更多有效监督信息以帮助检测分支的优化过程。VCM如下述公式及图4所示：

![image](https://user-images.githubusercontent.com/29652509/177359534-d5b08780-791e-4049-9c54-9748fe4fcb40.png)

![image](https://user-images.githubusercontent.com/29652509/177359549-2c1e85be-6683-4107-b478-4da8ac5d3dc7.png)

图4 视觉协作机制（VCM）

对于视觉富文档图像，视觉特征中集成了关键的视觉线索，例如形状，字体和颜色等等。信息抽取分支的梯度也可以帮助检测分支学习更泛化的有效表示。

##### 2）文本识别分支

该工作采用了类似传统基于注意力机制的文本识别网络结构，并提出了语义协作机制（SCM）以建立识别分支与信息抽取分支间的双向语义信息流。本文将识别分支中的循环神经网络隐状态S作为每个字符的高级语义表示：

![image](https://user-images.githubusercontent.com/29652509/177359728-affd43bb-fde2-4f08-97ff-4b9e41003aff.png)

同时，该工作还进一步生成段级别语义特征来融合更全局的信息。它采用1维CNN网络通过字符语义嵌入得到文本段的整体语义表达，其过程如下述公式及图5所示：

![image](https://user-images.githubusercontent.com/29652509/177359765-2a12c708-a429-4941-a966-ac46e24e2a93.png)

![image](https://user-images.githubusercontent.com/29652509/177359778-74f911b3-26f8-4ec7-8139-61dfcec169aa.png)

图5  语义协作机制（SCM）

通过这种方式，识别分支所提取的字符级和片段级语义信息可以直接向后传递，而信息抽取分支所包含的更高级语义约束也可以反过来指导识别分支的训练过程。

##### 3）信息抽取分支

在信息抽取模块，该工作首先通过检测到的文本框提取空间位置特征：

![image](https://user-images.githubusercontent.com/29652509/177359876-b442fb69-45e7-4ccd-8e4a-d00af9fecf64.png)

值得注意的是，本文根据识别出的字符串的长度将整个片段级别文本框沿最长边均匀地划分为多个单字符框，并以此利用上述提到的计算公式，可分别得到字符级别和片段级别的视觉和位置特征。
在得到来自多源的多细粒度特征表示后，本文提出自适应特征融合模块（AFFM）对信息执行进一步增强。AFFM由多头自注意力模块和线性变换层组成：

![image](https://user-images.githubusercontent.com/29652509/177359932-d2626fcb-cfdf-4101-a56d-449a188b258f.png)

最后，本文将字符级和片段级融合特征拼接在一起，送入最后的序列标注模型。本文采用经典的双向长短时记忆网络（BiLSTM）与条件随机场层（CRF layer）结构，对识别模块得到的结果进行分类。
整个框架在训练时可以进行端到端的联合优化，信息抽取部分的梯度可以回传至整个网络。整体的损失函数即由各子分支的优化目标加权构成。

### 四、主要实验结果及可视化效果

表2  端到端联合优化策略消融实验结果

![image](https://user-images.githubusercontent.com/29652509/177360012-fe3b1d07-9d38-4323-a9f0-221d7f2d56fd.png)

本文首先探究了其提出的端到端联合优化方式的有效性，实验结果如表2所示。端到端方法使模型在各子任务上的表现都有显著的提升。

表3  VCM与SCM结构消融实验结果

![image](https://user-images.githubusercontent.com/29652509/177360065-1cb010ba-0243-434b-aa66-ed147211e2b1.png)

接着，该工作对比了VCM和SCM不同建模方式的区别，实验结果如表3所示。本文最终选用的方式可以充分地发挥端到端优化的效果。

表4  不同来源特征消融实验结果

![image](https://user-images.githubusercontent.com/29652509/177360134-af623842-8c44-42bd-8bb5-6988bbe15b91.png)

最后，该工作同样探究了不同来源特征的影响，实验结果如表4所示。信息的模态多样性可以为模型效果带来进一步的提升。

表5  EPHOIE数据集实验结果

![image](https://user-images.githubusercontent.com/29652509/177360180-3d5ee49f-590d-4a77-a335-5988061b02e5.png)

表6  SROIE数据集实验结果

![image](https://user-images.githubusercontent.com/29652509/177360301-e89bf002-5e29-4280-9102-da2c5878fdd7.png)

 表5和表6展示了部分当下最先进方法在EPHOIE数据集和SROIE数据集上的结果。可以看到，本文提出的VIES取得了最高的指标。 
 
 ![image](https://user-images.githubusercontent.com/29652509/177360300-39d4a0f9-79c2-49f5-a858-3a1f2cdf5cfc.png)
 
 图6  EPHOIE数据集端到端结果的部分可视化
 
 图6展示了一些在EPHOIE数据集上的可视化结果。不同颜色代表提取出的不同实体类别。 
 
### 五、相关资源

论文及数据集地址：https://github.com/HCIILAB/EPHOIE

