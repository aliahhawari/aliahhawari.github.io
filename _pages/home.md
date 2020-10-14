---
layout: splash
permalink: /
hidden: true
header:
  overlay_color: "#5e616c"
  overlay_image: "/images/technologybanner.jpg"
  actions:
    - label: "<i class='fas fa-download'></i> Install now"
      url: "/docs/quick-start-guide/"
excerpt: >
  Welcome to my Data Science portfolio<br />
  <small><a href="https://aliahhawari.github.io/projects/">See full list of my projects here</a></small>
feature_row:
  - image_path: /images/DW/COVID/covid.jpg
    alt: "Python analysis"
    title: "COVID-19 analysis"
    excerpt: "Analysing the relationship between infection rates, death rates and the well being of countries"
    url: "https://aliahhawari.github.io/datawrangling/covid19analysis/"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: images/DW/handwashing/handwashing.jpg
    alt: "Photo by Sharon McCutcheon on Unsplash"
    title: "Wash your hands."
    excerpt: "Reanalysing the Dr. Semmelweis' dataset"
    url: "https://aliahhawari.github.io/datawrangling/handwashing/"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /images/KNN_images/pink-ribbon2.jpg
    alt: "Project 2"
    title: "Benign or Malignant?"
    excerpt: "Building a KNN classifier from Winconsin breast cancer data."
    url: "https://aliahhawari.github.io/machinelearning/Breast_cancer_KNN/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
feature_row1:
  - image_path: images/KNN_images/flower-iris2.jpg
    alt: "Iris project"
    title: "Verbosa, Setosa or Versicolor?"
    excerpt: "Using KNN Classifier to predict different species of Iris"
    url: "https://aliahhawari.github.io/machinelearning/KNN_iris"
    btn_label: "See More"
    btn_class: "btn--primary"
---


{% include feature_row %}

{% include feature_row id="feature_row1" type="right" %}
