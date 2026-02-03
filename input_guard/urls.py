from django.contrib import admin
from django.urls import path, include

from input_guard.views import InputEvaluatoreView

urlpatterns = [
    path('input_guardrail/',InputEvaluatoreView.as_view(), name='input_guardrail')
]

