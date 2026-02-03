from django.contrib import admin
from django.urls import path, include

from output_guard.views import OutputEvaluatoreView

urlpatterns = [
    path('output_evaluation/', OutputEvaluatoreView.as_view(), name='output_evaluation'),
]

