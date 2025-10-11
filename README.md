<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    max-width: 900px; /* 화면 폭 제한 */
    margin: 0 auto;   /* 전체 중앙 배치 */
    padding: 20px;
  }

  h1 {
    text-align: center;   /* 중앙 정렬 */
    margin-bottom: 20px;
    word-wrap: break-word; /* 길어도 줄바꿈 */
  }

  p {
    text-align: left;     /* 왼쪽 정렬 */
    margin-bottom: 20px;
  }
</style>
</head>
<body>

<h1>
  Official implementation of the paper "BAC-GCN: Background-Aware CLIP-GCN Framework for Unsupervised Multi-Label Classification" [ACM MM '25]
</h1>
<p>
  This is an official implementation of the ACM MM paper "BAC-GCN: Background-Aware CLIP-GCN Framework for Unsupervised Multi-Label Classification"
</p>

<h1>
  Abstract
</h1>
<p>
  Multi-label classification has recently demonstrated promising performance through CLIP-based unsupervised learning. However, existing CLIP-based approaches primarily focus on object-centric features, which limits their ability to capture rich contextual dependencies between objects and their surrounding scenes. In addition, the vision transformer architecture of CLIP exhibits a bias toward the most prominent object, often failing to recognize small or less conspicuous objects precisely. To address these limitations, we propose Background-Aware CLIP-GCN (BAC-GCN), a novel framework that explicitly models class-background interactions and is designed to capture fine-grained visual patterns of small objects effectively. BAC-GCN is composed of three key components: (i) a Similarity Kernel that extracts patch-level local features for each category (i.e., class and background), (ii) a CLIP-GCN that captures relational dependencies between local-global and class-background features, and (iii) a Re-Training for Small Objects (ReSO) strategy that enhances the representation of small and hard-to-learn objects by learning their distinctive visual characteristics. Therefore, our method facilitates a deeper understanding of complex visual contexts, enabling the model to make decisions by leveraging diverse visual cues and their contextual relationships. Extensive experiments demonstrate that BAC-GCN achieves state-of-the-art performance on three benchmark multi-label datasets: VOC07, COCO, and NUS, validating the effectiveness of our approach.
</p>

<h1>
  Main Result
</h1>
<p>
</p>

<h1>
  Result
</h1>
<p>
</p>

</body>
</html>
