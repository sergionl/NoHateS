from NoHateSAPI.viewsets import InfoViewset
from rest_framework import routers

router = routers.DefaultRouter()

router.register('info',InfoViewset)