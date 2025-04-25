---
author: [""]
title: "TIL this week" 
date: "2025-04-25"
tags: ["til"]
description: ""
summary: "Interesting things I learnt this week"
ShowToc: false
ShowBreadCrumbs: false
---

I listen to the [Everything is Everything](https://music.youtube.com/podcast/YwC6QmEsBbA) podcast by two cool dudes giving out lessons in general about life, education, politics, chess, poker, governments, you-name-it. In the latest podcast, they talk about globalization - 101, it's history, and some myths around tariffs.

* [Double 'Thank You' movement](https://abcnews.go.com/2020/story?id=3231572): Trade is a positive same game where both participating the parties benefit from the a particular transcation.

* Tariffs redistribute wealth from Poor to Rich: For example a competitor from another country is selling a product for $100 and the local producer is selling the same product for $110. Most would opt to a cheaper alternative. The local producer complain the government to protect domestic producers and slaps a tariff of $20 making the producer from other country cost $120. Now the local producer would become the cheaper alternative but consumers have to shelve $10 that would have been spent elsewhere.

---

Group Convolutions

There is an interesting parameter `groups` in [convolution](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) operation for PyTorch. It is useful in scenarios when we want to process each channel separately.

{{< figure src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*6MqiPb7UehJwBqJDkt77NA.png" attr="[Reference blog](https://medium.com/@jiahao.cao.zh/trying-out-pytorchs-group-convolution-83cc270cdfd)" align=center link="https://medium.com/@jiahao.cao.zh/trying-out-pytorchs-group-convolution-83cc270cdfd" target="_blank" >}}