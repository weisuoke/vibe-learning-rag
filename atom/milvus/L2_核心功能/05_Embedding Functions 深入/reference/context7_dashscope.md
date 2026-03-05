### POST /ocr.cn-shanghai.aliyuncs.com

Source: https://help.aliyun.com/zh/viapi/developer-reference/api-sy75xq

Recognizes character content and bounding box coordinates within an image. This API is suitable for multi-scenario image text recognition.

```APIDOC
## POST /ocr.cn-shanghai.aliyuncs.com

### Description
Recognizes character content and bounding box coordinates within an image. This API is suitable for multi-scenario image text recognition.

### Method
POST

### Endpoint
/ocr.cn-shanghai.aliyuncs.com

### Parameters
#### Query Parameters
- **Action** (String) - Required - System-defined parameter. Value: RecognizeCharacter.
- **ImageURL** (String) - Required - The URL of the image. It is recommended to use an OSS link in the Shanghai region. For files on local machines or OSS links in other regions, please refer to File URL processing.
- **MinHeight** (Integer) - Required - The minimum height of characters in the image, in pixels.
- **OutputProbability** (Boolean) - Required - Whether to output the probability of the text bounding box. Values:
  * true: Output the probability.
  * false: Do not output the probability.

### Request Example
```json
{
  "Action": "RecognizeCharacter",
  "ImageURL": "http://viapi-test.oss-cn-shanghai.aliyuncs.com/viapi-3.0domepic/ocr/RecognizeCharacter/RecognizeCharacter5.jpg",
  "MinHeight": 10,
  "OutputProbability": true
}
```

### Response
#### Success Response (200)
- **RequestId** (String) - The request ID.
- **Data** (Object) - The result data content.
  - **Results** (Array of Objects) - Recognition information.
    - **TextRectangles** (Object) - The position of the text bounding box.
      - **Top** (Integer) - The y-coordinate of the top-left corner of the text region.
      - **Width** (Integer) - The width of the text region.
      - **Height** (Integer) - The height of the text region.
      - **Angle** (Integer) - The angle of the text region. Range: `[-180, 180]`. Note: Rotation is around the center of the text region; positive for clockwise, negative for counter-clockwise.
      - **Left** (Integer) - The x-coordinate of the top-left corner of the text region.
    - **Text** (String) - The recognized text content.
    - **Probability** (Float) - The probability of the recognized text content. Range: 0 to 1.

#### Response Example (JSON)
```json
{
  "RequestId": "7A9BC7FE-2D42-57AF-93BC-09A229DD2F1D",
  "Data": {
    "Results": [
      {
        "TextRectangles": {
          "Left": 599,
          "Top": 160,
          "Angle": -69,
          "Height": 107,
          "Width": 26
        },
        "Probability": 0.99,
        "Text": "HAPPY"
      },
      {
        "TextRectangles": {
          "Left": 576,
          "Top": 150,
          "Angle": -63,
          "Height": 200,
          "Width": 37
        },
        "Probability": 0.99,
        "Text": "birthday"
      },
      {
        "TextRectangles": {
          "Left": 511,
          "Top": 150,
          "Angle": -65,
          "Height": 409,
          "Width": 77
        },
        "Probability": 0.99,
        "Text": "祝你生日快乐"
      }
    ]
  }
}
```
```

--------------------------------

### POST /RecognizePdf

Source: https://help.aliyun.com/zh/viapi/developer-reference/api-pdf-recognition

This endpoint allows you to recognize and extract text content from PDF documents. It supports providing the PDF file via a URL.

```APIDOC
## POST /RecognizePdf

### Description
This endpoint allows you to recognize and extract text content from PDF documents. It supports providing the PDF file via a URL.

### Method
POST

### Endpoint
`http(s)://ocr.cn-shanghai.aliyuncs.com/`

### Parameters
#### Query Parameters
- **Action** (string) - Required - The action to perform, should be "RecognizePdf".
- **FileURL** (string) - Required - The URL of the PDF file to process.
- **公共请求参数** (object) - Optional - Common request parameters for Aliyun APIs.

### Request Example
```http
http(s)://ocr.cn-shanghai.aliyuncs.com/?Action=RecognizePdf&FileURL=https://viapi-test.oss-cn-shanghai.aliyuncs.com/ocr/xxxx.pdf&公共请求参数
```

### Response
#### Success Response (200)
Responses can be in either XML or JSON format, containing extracted text information and bounding box coordinates.

#### Response Example (XML)
```xml
HTTP/1.1 200 OK
Content-Type:application/xml

<RecognizePdfResponse>
    <RequestId>CD9A9659-ABEE-4A7D-837F-9FDF40879A97</RequestId>
    <Data>
        <WordsInfo>
            <Word>天津增值税</Word>
            <Angle>-88</Angle>
            <X>514</X>
            <Positions>
                <X>397</X>
                <Y>45</Y>
            </Positions>
            <Positions>
                <X>662</X>
                <Y>52</Y>
            </Positions>
            <Positions>
                <X>661</X>
                <Y>82</Y>
            </Positions>
            <Positions>
                <X>396</X>
                <Y>75</Y>
            </Positions>
            <Y>-69</Y>
            <Height>265</Height>
            <Width>29</Width>
        </WordsInfo>
        <WordsInfo>
            <Word>普通发票</Word>
            <Angle>0</Angle>
            <X>678</X>
            <Positions>
                <X>678</X>
                <Y>48</Y>
            </Positions>
            <Positions>
                <X>824</X>
                <Y>47</Y>
            </Positions>
            <Positions>
                <X>824</X>
                <Y>76</Y>
            </Positions>
            <Positions>
                <X>678</X>
                <Y>77</Y>
            </Positions>
            <Y>48</Y>
            <Height>29</Height>
            <Width>146</Width>
        </WordsInfo>
        <WordsInfo>
            <Word>发票代码：012002000211</Word>
            <Angle>0</Angle>
            <X>863</X>
            <Positions>
                <X>863</X>
                <Y>46</Y>
            </Positions>
            <Positions>
                <X>1068</X>
                <Y>46</Y>
            </Positions>
            <Positions>
                <X>1068</X>
                <Y>62</Y>
            </Positions>
            <Positions>
                <X>863</X>
                <Y>62</Y>
            </Positions>
            <Y>46</Y>
            <Height>16</Height>
            <Width>205</Width>
        </WordsInfo>
        <OrgWidth>610</OrgWidth>
        <Angle>0</Angle>
        <OrgHeight>394</OrgHeight>
        <Height>788</Height>
        <PageIndex>1</PageIndex>
        <Width>1220</Width>
    </Data>
</RecognizePdfResponse>
```

#### Response Example (JSON)
```json
HTTP/1.1 200 OK
Content-Type:application/json

{
  "RequestId" : "CD9A9659-ABEE-4A7D-837F-9FDF40879A97",
  "Data" : {
    "WordsInfo" : [ {
      "Word" : "天津增值税",
      "Angle" : -88,
      "X" : 514,
      "Positions" : [ {
        "X" : 397,
        "Y" : 45
      }, {
        "X" : 662,
        "Y" : 52
      }, {
        "X" : 661,
        "Y" : 82
      }, {
        "X" : 396,
        "Y" : 75
      } ],
      "Y" : -69,
      "Height" : 265,
      "Width" : 29
    }, {
      "Word" : "普通发票",
      "Angle" : 0,
      "X" : 678,
      "Positions" : [ {
        "X" : 678,
        "Y" : 48
      }, {
        "X" : 824,
        "Y" : 47
      }, {
        "X" : 824,
        "Y" : 76
      }, {
        "X" : 678,
        "Y" : 77
      } ],
      "Y" : 48,
      "Height" : 29,
      "Width" : 146
    }, {
      "Word" : "发票代码：012002000211",
      "Angle" : 0,
      "X" : 863,
      "Positions" : [ {
        "X" : 863,
        "Y" : 46
      }, {
        "X" : 1068,
        "Y" : 46
      }, {
        "X" : 1068,
        "Y" : 62
      }, {
        "X" : 863,
        "Y" : 62
      } ],
      "Y" : 46,
      "Height" : 16,
      "Width" : 205
    } ],
    "OrgWidth" : 610,
    "Angle" : 0,
    "OrgHeight" : 394,
    "Height" : 788,
    "PageIndex" : 1,
    "Width" : 1220
  }
}
```

### Error Handling
Refer to the common error codes for PDF recognition errors. [https://help.aliyun.com/document_detail/143103.html]
```

--------------------------------

### POST /RecognizePdf

Source: https://help.aliyun.com/zh/viapi/developer-reference/api-pdf-recognition

The RecognizePdf API enables structured text recognition from PDF documents. It processes PDF files and returns the extracted text along with its structural and positional information.

```APIDOC
## POST /RecognizePdf

### Description
Extracts structured text from PDF documents. This API is useful for applications requiring automated data extraction from PDFs, such as content review and business expense processing.

### Method
POST

### Endpoint
/RecognizePdf

### Parameters
#### Query Parameters

#### Request Body
- **Action** (String) - Required - The system-defined value for this API, which is `RecognizePdf`.
- **FileURL** (String) - Required - The URL of the PDF file to be recognized. It is recommended to use an OSS link from the Shanghai region. Special handling is required for local files or OSS links from other regions.

### Request Example
```json
{
  "Action": "RecognizePdf",
  "FileURL": "https://viapi-test.oss-cn-shanghai.aliyuncs.com/ocr/xxxx.pdf"
}
```

### Response
#### Success Response (200)
- **RequestId** (String) - The unique ID of the request.
- **Data** (Object) - Contains the results of the PDF recognition.
  - **Height** (Long) - The height of the document after rotation.
  - **Width** (Long) - The width of the document after rotation.
  - **OrgHeight** (Long) - The original height of the document.
  - **OrgWidth** (Long) - The original width of the document.
  - **PageIndex** (Long) - The number of pages in the PDF.
  - **Angle** (Long) - The rotation angle of the PDF file.
  - **WordsInfo** (Array of objects) - An array containing information about each recognized word.
    - **Angle** (Long) - The rotation angle of the recognized field.
    - **Word** (String) - The recognized text content.
    - **Height** (Long) - The height of the bounding box for the recognized word.
    - **Width** (Long) - The width of the bounding box for the recognized word.
    - **X** (Long) - The X-coordinate of the top-left corner of the bounding box.
    - **Y** (Long) - The Y-coordinate of the top-left corner of the bounding box.
    - **Positions** (Array of objects) - The coordinates of the bounding box corners in clockwise order (top-left, top-right, bottom-right, bottom-left).
      - **X** (Long) - The X-coordinate of a corner.
      - **Y** (Long) - The Y-coordinate of a corner.

#### Response Example
```json
{
  "RequestId": "CD9A9659-ABEE-4A7D-837F-9FDF40879A97",
  "Data": {
    "Height": 788,
    "Width": 1220,
    "OrgHeight": 610,
    "OrgWidth": 394,
    "PageIndex": 1,
    "Angle": 0,
    "WordsInfo": [
      {
        "Angle": 0,
        "Word": "发票代码：012002000211",
        "Height": 16,
        "Width": 205,
        "X": 863,
        "Y": 46,
        "Positions": [
          {"X": 863, "Y": 43},
          {"X": 1068, "Y": 43},
          {"X": 1068, "Y": 62},
          {"X": 863, "Y": 62}
        ]
      }
    ]
  }
}
```

### Input Restrictions
- **File Format**: PDF
- **File Size**: Maximum 10 MB
- **Document Length**: Maximum 5 pages for PDF
- **URL**: URL addresses must not contain Chinese characters.
```

Source: https://help.aliyun.com/zh/viapi/developer-reference/api-sy75xq

通用文字识别 > 返回数据: 名称| 类型| 示例值| 描述 | 名称 | 类型 | 示例值 | 描述  
---|---|---|---  
RequestId | String | 7A9BC7FE-2D42-57AF-93BC-09A229DD2F1D | 请求ID。  
Data | Object |  | 返回的结果数据内容。  
Results | Array of Result |  | 返回识别信息。  
TextRectangles | Object |  | 文字框区域位置。  
Top | Integer | 150 | 文字区域左上角y坐标。  
Width | Integer | 77 | 文字区域宽度。  
Height | Integer | 409 | 文字区域高度。  
Angle | Integer | -65 | 文字区域角度，角度范围`[-180, 180]`。 __**说明** 以文字区域中心点为旋转点，向右旋转角度为正，向左旋转角度为负。  
Left | Integer | 511 | 文字区域左上角x坐标。  
Text | String | 祝你生日快乐 | 文字内容。  
Probability | Float | 0.99 | 文字内容的概率，取值范围为0~1。

--------------------------------

Source: https://help.aliyun.com/zh/viapi/developer-reference/api-video-text-recognition

视频文字识别 > 返回数据: **名称** | **类型** | **示例值** | **描述**  
---|---|---|---  
**名称** | **类型** | **示例值** | **描述**  
---|---|---|---  
RequestId | String | D3F5BA69-79C4-46A4-B02B-58C4EEBC4C33 | 请求ID。  
Data | Object |  | 返回的结果数据内容。 该数据需要在异步任务执行成功后，通过调用GetAsyncJobResult接口，对其Result字段进行JSON反序列化之后得到。  
Width | Long | 1920 | 视频宽度分辨率，单位像素。  
Height | Long | 1080 | 视频高度分辨率，单位像素。  
Frames | Array of Frame |  | 视频帧的集合，空信息的帧不展示。  
Timestamp | Long | 6124533574 | 帧时间戳，单位毫秒。  
Elements | Array of Element |  | 文字区域元素列表  
Score | Float | 0.99 | 文字区域概率，概率值的范围为[0.0,1.0]。  
Text | String | 在桃花盛开的地方 | 文字内容。  
TextRectangles | Array of TextRectangle |  | 文字区域位置信息。  
Angle | Long | -90 | 文字区域角度，角度范围[-180, 180]。  
Left | Long | 213 | 文字区域左上角X坐标。  
Top | Long | 98 | 文字区域左上角Y坐标。  
Width | Long | 46 | 文字区域宽度，单位像素。  
Height | Long | 213 | 文字区域高度，单位像素。  
InputFile | String | oss://my-bucket/a/b/c.mp4 | 输入视频文件OSS地址。  
Message | String | 该调用为异步调用，任务已提交成功，请以requestId的值作为jobId参数调用同类目下GetAsyncJobResult接口查询任务执行状态和结果。 | 提交异步任务后的提示信息。