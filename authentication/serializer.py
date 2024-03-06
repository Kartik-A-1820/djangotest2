from django.contrib.auth import authenticate
from rest_framework import serializers
import pickle
from .models import User
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

class SignupSerializer(serializers.Serializer):
    email = serializers.EmailField()
    username = serializers.CharField()
    password = serializers.CharField()
    phone_number = serializers.CharField(required=False)

    def validate(self, data):
        email = data.get('email')
        phone_number = data.get('phone_number')

        if not email.strip() and not phone_number.strip():
            raise serializers.ValidationError("Either email or phone number must be provided.")
        if not email.strip():
            raise serializers.ValidationError("Email must be provided.")
        return data

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user
    
class OTPVerificationSerializer(serializers.Serializer):
    otp = serializers.CharField(max_length=6)