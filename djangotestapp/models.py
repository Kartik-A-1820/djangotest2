from django.db import models
# Global settings
BASE_DIR = r"C:/Users/hp/Desktop/MobileAPI/djangotest/djangotestapp"
# LOG_DIR = f"{BASE_DIR}log/"
# Create your models here.
class Results(models.Model):
    serial_number = models.AutoField(primary_key=True)  # Auto-generated serial number
    species_name = models.CharField(max_length=100)
    input_image_url = models.URLField()  # Store input image URL from S3
    output_image_url = models.URLField()  # Store output/result image URL from S3
    timestamp = models.DateTimeField(auto_now_add=True)
    total_count = models.PositiveIntegerField(default=0)
    good_count = models.PositiveIntegerField(default=0)
    bad_count = models.PositiveIntegerField(default=0)
    prediction_reason = models.CharField(max_length=100, null=True, blank=True, default=None)
    user_name = models.CharField(max_length=100, null=True, blank=True, default=None)

    class Meta:
        db_table = 'Results'

    def __str__(self):
        return f"{self.serial_number}: {self.species_name}"


class Feedback(models.Model):
    serial_number = models.AutoField(primary_key=True)
    input_image_url = models.URLField()  # Store input image URL from S3
    output_image_url = models.URLField()  # Store output/result image URL from S3
    fish1_actual = models.CharField(max_length=1, choices=[('G', 'Good'), ('B', 'Bad')], null=True)
    fish2_actual = models.CharField(max_length=1, choices=[('G', 'Good'), ('B', 'Bad')], null=True)
    fish3_actual = models.CharField(max_length=1, choices=[('G', 'Good'), ('B', 'Bad')], null=True)
    fish1_pred = models.CharField(max_length=1, choices=[('G', 'Good'), ('B', 'Bad')], null=True)
    fish2_pred = models.CharField(max_length=1, choices=[('G', 'Good'), ('B', 'Bad')], null=True)
    fish3_pred = models.CharField(max_length=1, choices=[('G', 'Good'), ('B', 'Bad')], null=True)
    result = models.CharField(max_length=7, choices=[('Correct', 'Correct'), ('False', 'False')])
    prediction_reason = models.CharField(max_length=100, null=True, blank=True, default=None)
    timestamp = models.DateTimeField(auto_now_add=True)
    user_name = models.CharField(max_length=100, null=True, blank=True, default=None)

    class Meta:
        db_table = 'Feedback'

    def save(self, *args, **kwargs):
        if (
            self.fish1_actual == self.fish1_pred and
            self.fish2_actual == self.fish2_pred and
            self.fish3_actual == self.fish3_pred
        ):
            self.result = 'Correct'
        else:
            self.result = 'False'
        super().save(*args, **kwargs)


    def __str__(self):
        return f"{self.serial_number}: Feedback"
