from django.contrib.auth import authenticate
from rest_framework import serializers

#class LoginSerializer(serializers.Serializer):
#    email = serializers.CharField()
#    password = serializers.CharField()
#
#    def validate(self, data):
#        user = authenticate(email=data['email'], password=data['password'])
#        if user:
#            data['user'] = user
#        else:
#            raise serializers.ValidationError("Invalid login credentials")
#        return data
import pickle

class LoginSerializer(serializers.Serializer):
    email = serializers.CharField()
    password = serializers.CharField()

    def validate(self, data):
        try:
            with open(r'/home/ec2-user/DJANGOTEST/djangotestapp/user.pkl', 'wb') as file:
                pickle.dump(data['email'], file=file)
        except:
            with open(r'djangotestapp/user.pkl', 'wb') as file:
                pickle.dump(data['email'], file=file)
        user = authenticate(email=data['email'], password=data['password'])
        if user:
            data['user'] = user
        else:
            raise serializers.ValidationError("Invalid login credentials")
        return data
