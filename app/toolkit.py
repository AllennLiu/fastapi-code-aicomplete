from typing import Dict
from textwrap import dedent

class Tools:
    tools = [
        {
            "name"       : "get_current_weather_by_location",
            "description": "根据城市获取当前天气",
            "parameters" : {
                "type"      : "object",
                "properties": { "location": { "description": "城市名称 e.g. 北京，上海，武汉" } },
                "required"  : [ 'location' ]
            }
        },
        {
            "name"       : "get_news_about_birthday",
            "description": "获取生日庆祝相关的新闻文章",
            "parameters" : {}
        }
    ]

    @staticmethod
    def get_current_weather_by_location(location: str) -> Dict[str, str]:
        """根据城市获取当前天气"""
        print("天气函数被调用了")
        if location == "北京":
            return {
                "location": location,
                "temp": "24",
                "text": "多云",
                "windDir": "东南风",
                "windScale": "1"
            }
        elif location == "武汉":
            return {
                "location": location,
                "temp": "26",
                "text": "晴",
                "windDir": "西北风",
                "windScale": "2"
            }

    @staticmethod
    def get_news_about_birthday() -> Dict[str, str]:
        """获取生日庆祝相关的新闻文章"""
        return {
            "content": dedent("""\
                参加朋友的生日会，你也一起出主意、构想庆祝内容，是不是会更有参与感、更难忘？
                学前儿童刊物《小小拇指》为庆祝创刊10周年，将在6月至8月的巡回展上，举办多场免费的华语讲故事活动。这次讲故事活动的最大特色是将以参与式剧场的形式，带领孩童和家长一同参加“小拇指的生日会”。这种形式鼓励参与者积极发挥创意、贡献点子，每个人都是故事的创建者，参与越多，融入感就越强。
                主讲者符妙娟（39岁，戏剧工作者）与同伴林慈暄（协助执行）用了一个月的时间，构思故事脚本和互动方式。故事带领小朋友回顾10年来《小小拇指》的重要内容，比如认识新加坡、各种新鲜趣闻，以及朗朗上口的本土儿歌，过程中会穿插各种想法交流和亲自动手的环节。
                """)
        }
