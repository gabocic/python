from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
from django.views.decorators.csrf import csrf_protect
from django.db.models import Q
import json
from .models import Parameter
from .forms import ParameterFormSet
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db import transaction
from django.forms import inlineformset_factory



def parametermanager(request):
    success = 0
    if request.method == 'GET':
        formset = ParameterFormSet(initial=Parameter.objects.all().values())
    elif request.method == 'POST':
        formset = ParameterFormSet(request.POST)
        if formset.is_valid() and formset.has_changed():
            for form in formset:
                try:
                    form.save(['parameter_value'])
                except:
                    success = -1
                else:
                    success = 1
        elif formset.has_changed() == False:
            success = 2
    context = {'formset':formset,'success':success}
    return render(request,'myapp/parameter.html',context)
