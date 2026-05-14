---
layout: reading
title: reading
permalink: /reading/
nav: true
nav_order: 6
collection: reading
---

{% assign reading_items = site.reading | sort: 'category' | reverse %}

{% for item in reading_items %}
  <h3>{{ item.title }}</h3>
  <p>{{ item.content }}</p>
{% endfor %}