---
title: "Machine Learning"
layout: splash
permalink: /machinelearning/
header:
  overlay_color: "#012"
  overlay_filter: "0.5"
  overlay_image: /images/bokeh.jpg
  actions:
    - label: "Download"
      url: "https://github.com/mmistakes/minimal-mistakes/"
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
excerpt: "A collection of various data science projects I have undertaken."
feature_row:
  - image_path: images/KNN_images/flower-iris2.jpg
    alt: "Iris project"
    title: "Predicting the species of Iris flowers"
    excerpt: "Using a KNN classifier, the species of Iris could be predicted based on its sepal and petal measurements"
    url: "https://aliahhawari.github.io/machinelearning/KNN_iris"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /images/KNN_images/pink-ribbon2.jpg
    alt: "Breast cancer project"
    title: "Is the tumour benign or malignant?"
    excerpt: "Classifying breast cancer data of patients from Winsconsin using KNN."
    url: "https://aliahhawari.github.io/machinelearning/Breast_cancer_KNN"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /assets/images/unsplash-gallery-image-3-th.jpg
    title: "Placeholder 3"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
feature_row2:
  - image_path: assets/images/unsplash-gallery-image-1-th.jpg
    alt: "placeholder image 1"
    title: "Machine Learning"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
    url: "https://aliahhawari.github.io/foo/"
    btn_label: "See More"
    btn_class: "btn--primary"
  - image_path: /assets/images/unsplash-gallery-image-2-th.jpg
    image_caption: "Image courtesy of [Unsplash](https://unsplash.com/)"
    alt: "placeholder image 2"
    title: "Data Wrangling"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
    url: "https://aliahhawari.github.io/foo/"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /assets/images/unsplash-gallery-image-3-th.jpg
    title: "Placeholder 3"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
---

{% include feature_row id="intro" type="center" %}

{% include feature_row %}

{% include feature_row id="feature_row2" type="left" %}

{% include feature_row id="feature_row3" type="right" %}

{% include feature_row id="feature_row4" type="center" %}
