import requests
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
import pickle

def update_inspection_result(InspectionLot, InspPlanOperationInternalID, InspectionCharacteristic, patch_data, ip_address='34.77.141.38'):
    username = 'QZENSE1'
    password = 'Qzense@123'
    url_template = f"http://{ip_address}:50000/sap/opu/odata/sap/API_INSPECTIONLOT_SRV/A_InspectionResult(InspectionLot='{InspectionLot}',InspPlanOperationInternalID='{InspPlanOperationInternalID}',InspectionCharacteristic='{InspectionCharacteristic}')"

    auth = (username, password)
    headers = {
              'Accept': 'application/json',
              'x-csrf-token':'Fetch'
               }
    response = requests.get(url_template, auth=auth, headers=headers)

    if response.status_code != 200:
        print(f"Error: Unable to retrieve data for InspectionLot '{InspectionLot}'")
        return

    Cookie = response.headers.get('set-cookie')
    Cookie = Cookie.split(',')
    Cookie = ';'.join(Cookie)

    headers_patch = {
        'Accept': 'application/json',
        'x-csrf-token': response.headers.get('x-csrf-token'),
        'If-Match': response.headers.get('etag'),
        'Cookie': Cookie,
    }

    response_p = requests.patch(url_template, auth=auth, headers=headers_patch, json=patch_data, cookies=response.cookies)

    return response_p


def is_supported(InspectionLot, InspPlanOperationInternalID, InspectionCharacteristic):
    username = 'QZENSE1'
    password = 'Qzense@123'
    auth = (username, password)
    headers = {
    'Accept': 'application/json',
    'x-csrf-token': 'Fetch'
    }
    url2 = f"http://34.77.141.38:50000/sap/opu/odata/sap/API_INSPECTIONLOT_SRV/A_InspectionResult(InspectionLot='{InspectionLot}',InspPlanOperationInternalID='{InspPlanOperationInternalID}',InspectionCharacteristic='{InspectionCharacteristic}')"
    response = requests.get(url2, auth=auth, headers=headers)
    etag = response.headers['etag']
    csrf = response.headers['x-csrf-token']
    Cookie = response.headers['set-cookie']
    Cookie = Cookie.split(',')
    Cookie = ';'.join(Cookie)
    cookies = response.cookies
    headersp = {
                'Accept': 'application/json',
                'x-csrf-token': csrf,
                'If-Match': etag,
                'Cookie':Cookie
            }
    patch_data = {
        'Inspector': 'Test'
    }
    response_p = requests.patch(url2, auth=(username, password), headers=headersp, json=patch_data, cookies=cookies)
    if response_p.status_code == 204 or response_p.status_code == 200:
      return True
    else:
      return False



def get_lot_list():
    update_list = []
    username = 'QZENSE1'
    password = 'Qzense@123'
    url_template = f"http://34.77.141.38:50000/sap/opu/odata/sap/API_INSPECTIONLOT_SRV/A_InspectionResult?$filter=InspectionCharacteristic eq '20'"
    auth = (username, password)
    headers = {
    'Accept': 'application/json'
    }
    response = requests.get(url_template, auth=auth, headers=headers)
    lots = []
    for lot in response.json()['d']['results']:
        InspectionLot=lot['InspectionLot']
        InspPlanOperationInternalID=lot['InspPlanOperationInternalID']
        InspectionCharacteristic = lot['InspectionCharacteristic']
        if InspPlanOperationInternalID == "1":
            lots.append(list([InspectionLot, InspPlanOperationInternalID, InspectionCharacteristic]))
    for lot in lots:
        if is_supported(lot[0],lot[1], lot[2]):
            update_list.append(lot)
    return update_list  


class postDataSapView(APIView):
    def get(self, request):
        if request.method == "GET":
            my_list = get_lot_list()
        if len(my_list) > 0:
            return Response({'Message':'Update Successful', 'List':my_list }, status=status.HTTP_200_OK)
        else:
            return Response({'Message':'Fetching Data Failed!'}, status=status.HTTP_400_BAD_REQUEST)
        
    def post(self, request):
        if request.method=="POST":
            form = request.POST
            InspectionLot = form['InspectionLot']
            InspPlanOperationInternalID = form['InspPlanOperationInternalID']
            InspectionCharacteristic = form['InspectionCharacteristic']
            Inspector = "Qzense"
            InspectionValuationResult= form['InspectionValuationResult']
            try:
                with open('djangotestapp/result.pkl', 'rb') as file:
                    result = pickle.load(file)
            except:
                result = None
            if result:
                good = result['good-fishes']
                bad = result['bad-fishes']
                total = good+bad
            else:
                good = 0
                bad = 0
                total = 0

            InspectionResultText = f"Total:{total} Good:{good} Bad:{bad}"

            patch_data = {
                'InspectionLot':InspectionLot,
                'InspPlanOperationInternalID':InspPlanOperationInternalID,
                'InspectionCharacteristic':InspectionCharacteristic,
                'Inspector':Inspector,
                'InspectionValuationResult':InspectionValuationResult,
                'InspectionResultText':InspectionResultText,
                'InspectionResultStatus' : '5',
            }

            r = update_inspection_result(InspectionLot=InspectionLot,InspPlanOperationInternalID=InspPlanOperationInternalID, InspectionCharacteristic=InspectionCharacteristic, patch_data=patch_data)
            code = r.status_code
            if code == 204 or code == 200:
                return Response({'Message':'Update Successful'}, status=status.HTTP_200_OK)
            else:
                return Response({'Message':'Update Failed!','Code':code, 'error':r.content}, status=status.HTTP_400_BAD_REQUEST)

