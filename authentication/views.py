from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from .serializer import LoginSerializer
from rest_framework_simplejwt.tokens import RefreshToken
import boto3
import random
from rest_framework import generics, status
from rest_framework.response import Response
from .models import OTP, User
from .serializer import SignupSerializer, OTPVerificationSerializer
from pathlib import Path 
import pickle
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
# Generate Token Manually
def get_tokens_for_user(user):
  refresh = RefreshToken.for_user(user)
  return {
      'refresh': str(refresh),
      'access': str(refresh.access_token),
  }
  
class LoginView(generics.GenericAPIView):
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token = get_tokens_for_user(user)
        return Response({'token': token,
                         'message':'Login Success'}, status = status.HTTP_200_OK)
    
class SignupView(generics.GenericAPIView):
    serializer_class = SignupSerializer

    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data.get('email')
        if User.objects.filter(email=email).exists():
            return Response({'message': 'User with this email already exists'}, status=status.HTTP_400_BAD_REQUEST)
        # Generate OTP
        otp = ''.join([str(random.randint(0, 9)) for _ in range(6)])

        # Save the serializer data to access it later
        serializer_data = serializer.validated_data
        serializer_data['otp'] = otp
        signup_data_path = Path(__file__).resolve().parent / 'data.pkl'
        with open(signup_data_path, 'wb') as f:
            pickle.dump(serializer_data, f)

        # Send OTP
        if serializer_data.get('phone_number'):
            self.send_otp_sms(serializer_data['phone_number'], otp)
            return Response({'message': f'OTP sent. Please verify your phone:{serializer_data["phone_number"]} with OTP'}, status=status.HTTP_201_CREATED)
        self.send_otp(serializer_data['email'], otp)

        return Response({'message': 'OTP sent. Please verify your email/phone with OTP to complete signup!', 'otp': otp, 'serializer_data': serializer_data}, status=status.HTTP_201_CREATED)

    def create_otp(self, user, otp):
        OTP.objects.create(user=user, otp=otp)

    def send_otp_sms(self, phone_number, otp):
        # Initialize AWS SNS client
        sns_client = boto3.client(
            'sns',
            aws_access_key_id='AKIA2QMGQOEAXYO3BIYV',
            aws_secret_access_key='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',
            region_name='ap-south-1',
        )

        # Send OTP via SMS to the phone number
        try:
          message = f"Your OTP for signup is: {otp}"
          response = sns_client.publish(
            PhoneNumber=phone_number,
            Message=message
          )
        except Exception as e:
            print(e)
        return response
        
    def send_otp(self, email, otp):
        # Initialize AWS SNS client
        sns_client = boto3.client(
            'sns',
            aws_access_key_id='AKIA2QMGQOEAXYO3BIYV',
            aws_secret_access_key='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',
            region_name='ap-south-1',
        )

        # Publish OTP to the email address
        response = sns_client.publish(
            TopicArn='your_sns_topic_arn',
            Message=f'Your OTP for signup is: {otp}',
            Subject='OTP Verification',
            MessageStructure='string',
            MessageAttributes={
                'email': {
                    'DataType': 'String',
                    'StringValue': email,
                }
            }
        )


class OTPVerificationView(generics.GenericAPIView):
    serializer_class = OTPVerificationSerializer

    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        otp = serializer.validated_data['otp']

        # Load the serialized signup data from the file
        signup_data_path = Path(__file__).resolve().parent / 'data.pkl'
        with open(signup_data_path, 'rb') as file:
            signup_data = pickle.load(file)

        # Check if the OTP from the signup data matches the input OTP
        if signup_data['otp'] != otp:
            return Response({'error': 'Invalid OTP'}, status=status.HTTP_400_BAD_REQUEST)

        # Create the user using the signup data (excluding the OTP)
        user = User.objects.create_user(email=signup_data['email'], username=signup_data['username'], password=signup_data['password'])
        # Additional user creation steps if needed
        if signup_data_path.exists():
            signup_data_path.unlink()
        return Response({'message': 'User created successfully'}, status=status.HTTP_201_CREATED)
    

class UserDeleteView(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    def post(self, request):
        email = request.data.get('email')
        try:
            user = User.objects.get(email=email)
            if request.user.is_admin:
                user.delete()
                return Response({'message': 'User deleted successfully'}, status=status.HTTP_204_NO_CONTENT)
            elif request.user == user:
                user.delete()
                return Response({'message': 'User deleted successfully'}, status=status.HTTP_204_NO_CONTENT)
            else:
                return Response({'error': 'You are not authorized to delete this user'}, status=status.HTTP_403_FORBIDDEN)
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)