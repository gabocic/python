from django import forms
from django.forms import ModelForm
from django.forms import BaseFormSet
from django.forms import modelformset_factory
from django.forms import ValidationError

from myapp.models import Parameter
from django.forms import inlineformset_factory

class ParameterForm(ModelForm):
    class Meta:
        model = Parameter
        ## Note the 'id' field..
        fields = ['id','parameter_name','parameter_value']
        widgets = {
                'parameter_name': forms.TextInput(attrs={'readonly':'readonly','style':'background-color:#e8f4fd','size':'30'}),
                }

    # Parameter additional validations
    def clean_parameter_value(self):
        pos_int_parameters = [
                'CHUNK_SIZE',
                'CON_TIMEOUT',
                'POL_INTERVAL',
                'SSH_PORT'
                ]
        if self.changed_data:
            parameter_name = self.cleaned_data.get('parameter_name')

            ## Validate that numeric parameters are integers and greater than 0
            if parameter_name in pos_int_parameters:
                try:
                    a = int(self.cleaned_data.get('parameter_value').__str__())
                except:
                    raise ValidationError("This value should be an integer")
                else:
                    if a < 1:
                        raise ValidationError("This value should be greater than 0")

            ## Check that the SSH port is within TCP/UDP port range
            if parameter_name in 'SSH_PORT':
                port = self.cleaned_data.get('parameter_value') 
                if int(port) > 65535:
                    raise ValidationError("This value should be between 1 and 65535")
        return self.cleaned_data.get('parameter_value')

ParameterFormSet = modelformset_factory(Parameter,ParameterForm,extra=0)
