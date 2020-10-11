---
title: "Projects"
layout: splash
permalink: /projects/
header:
  overlay_color: "#000"
  overlay_filter: "0.25"
  overlay_image: /images/projects.jpg
  actions:
    - label: "Download"
      url: "https://github.com/mmistakes/minimal-mistakes/"
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
excerpt: "My collection of various data science projects."
intro: 
  - excerpt: 'Nullam suscipit et nam, tellus velit pellentesque at malesuada, enim eaque. Quis nulla, netus tempor in diam gravida tincidunt, *proin faucibus* voluptate felis id sollicitudin. Centered with `type="center"`'
feature_row:
  - image_path: images/machinelearning.png
    alt: "Machinelearning"
    title: "Machine Learning"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
    url: "https://aliahhawari.github.io/machinelearning/"
    btn_label: "See More"
    btn_class: "btn--primary"
  - image_path: /images/teaser_images/dashboard.jpg
    image_caption: ""
    alt: "Exploratory data analysis"
    title: "Data Wrangling"
    excerpt: "Exploratiry data analysis projects"
    url: "https://aliahhawari.github.io/foo/"
    btn_label: "See More"
    btn_class: "btn--primary"
  - image_path: /images/teaser_images/library.jpg
    title: "Placeholder 3"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
feature_row2:
  - image_path: /assets/images/unsplash-gallery-image-2-th.jpg
    alt: "placeholder image 2"
    title: "Placeholder Image Left Aligned"
    excerpt: 'This is some sample content that goes here with **Markdown** formatting. Left aligned with `type="left"`'
    url: "#test-link"
    btn_label: "Read More"
    btn_class: "btn--primary"
feature_row3:
  - image_path: /assets/images/unsplash-gallery-image-2-th.jpg
    alt: "placeholder image 2"
    title: "Placeholder Image Right Aligned"
    excerpt: 'This is some sample content that goes here with **Markdown** formatting. Right aligned with `type="right"`'
    url: "#test-link"
    btn_label: "Read More"
    btn_class: "btn--primary"
feature_row4:
  - image_path: /assets/images/unsplash-gallery-image-2-th.jpg
    alt: "placeholder image 2"
    title: "Placeholder Image Center Aligned"
    excerpt: 'This is some sample content that goes here with **Markdown** formatting. Centered with `type="center"`'
    url: "#test-link"
    btn_label: "Read More"
    btn_class: "btn--primary"
---

{% include feature_row id="intro" type="center" %}

{% include feature_row %}

{% include feature_row id="feature_row2" type="left" %}

{% include feature_row id="feature_row3" type="right" %}

{% include feature_row id="feature_row4" type="center" %}

