from django.db import models
from django.contrib.auth.models import BaseUserManager,AbstractUser, AbstractBaseUser

# class User(AbstractUser):
#     USERNAME_FIELD = 'email'
#     email = models.EmailField(('email address'), unique=True) # changes email to unique and blank to false
#     REQUIRED_FIELDS = [] # removes email from REQUIRED_FIELDS
    
# class Profile(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE)
    
class UserManager(BaseUserManager):
    def create_user(self, username, email, password=None, phone_number=None):
        """
        Creates and saves a User with the given email, username, password, and phone_number.
        """
        if not email:
            raise ValueError('Users must have an email address')

        user = self.model(
            email=self.normalize_email(email),
            username=username,
            phone_number=phone_number  # Include phone_number argument here
        )

        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, email, password=None):
        user = self.create_user(
            email=email,
            password=password,
            username=username
        )
        user.is_admin = True
        user.save(using=self._db)
        return user



class User(AbstractBaseUser):
    email = models.EmailField(
        verbose_name='email address',
        max_length=255,
        unique=True,
    )
    username = models.CharField(max_length=255)
    phone_number = models.CharField(max_length=15, blank=True, null=True)  # Add phone_number field

    is_admin = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email

    def has_perm(self, perm, obj=None):
        "Does the user have a specific permission?"
        return self.is_admin

    def has_module_perms(self, app_label):
        "Does the user have permissions to view the app `app_label`?"
        return True

    @property
    def is_staff(self):
        "Is the user a member of staff?"
        return self.is_admin

class OTP(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    otp = models.CharField(max_length=6)
    is_verified = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.user.email} - {self.otp}'