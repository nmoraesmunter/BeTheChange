from unittest import TestCase
from targets_processor import TargetsProcessor

targets = [
    {
        "id": 750149,
        "display_name": "Mike Honda",
        "email": None,
        "type": "Politician",
        "slug": "mike-honda",
        "description": "Mike Honda proudly represents California 17th Congressional District in the U.S. House of Representatives. His district includes Silicon Valley, the birthplace of innovation and the national leader in high-tech development.",
        "publicly_visible": True,
        "verified_at": "2013-10-21T21:41:13Z",
        "summary": "US House of Representatives - California-17",
        "locale": "en-US",
        "confirmed_at": "2013-11-14T23:16:31Z",
        "is_person": True,
        "member_of": {},
        "additional_data": {
            "state": "CA",
            "title": "State Representative",
            "district": "17",
            "active": True
        },
        "photo": {
            "id": 11499763,
            "url": "photos/7/su/yh/DYSuYHtdzhjvHqa-fullsize.jpg",
            "sizes": {
                "small": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/su/yh/DYSuYHtdzhjvHqa-48x48-noPad.jpg?1423780512",
                    "processing": False,
                    "size": {
                        "width": 48,
                        "height": 48
                    }
                },
                "medium": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/su/yh/DYSuYHtdzhjvHqa-128x128-noPad.jpg?1423780513",
                    "processing": False,
                    "size": {
                        "width": 128,
                        "height": 128
                    }
                },
                "large": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/su/yh/DYSuYHtdzhjvHqa-400x400-noPad.jpg?1423780514",
                    "processing": False,
                    "size": {
                        "width": 400,
                        "height": 400
                    }
                },
                "xlarge": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/su/yh/DYSuYHtdzhjvHqa-800x800-noPad.jpg?1453772907",
                    "processing": False,
                    "size": {
                        "width": 800,
                        "height": 800
                    }
                }
            }
        }
    },
    {
        "id": 750505,
        "display_name": "Paul Ryan",
        "email": None,
        "type": "Politician",
        "slug": "paul-ryan",
        "description": "Paul Davis Ryan is the United States Representative for Wisconsin's 1st congressional district and current chairman of the House Budget Committee. He was the Republican Party nominee for Vice President of the United States in the 2012 election.",
        "publicly_visible": True,
        "verified_at": "2013-09-24T19:28:43Z",
        "summary": "US House of Representatives - Wisconsin-01",
        "locale": "en-US",
        "confirmed_at": "2013-11-14T23:16:31Z",
        "is_person": True,
        "member_of": {},
        "additional_data": {
            "state": "WI",
            "title": "Representative",
            "district": "01",
            "active": True
        },
        "photo": {
            "id": 138371210,
            "url": "photos/6/lm/yu/gqLmyuzMeoNfvBq-fullsize.jpg",
            "sizes": {
                "small": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/6/lm/yu/gqLmyuzMeoNfvBq-48x48-noPad.jpg?1464387230",
                    "processing": False,
                    "size": {
                        "width": 48,
                        "height": 48
                    }
                },
                "medium": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/6/lm/yu/gqLmyuzMeoNfvBq-128x128-noPad.jpg?1464387231",
                    "processing": False,
                    "size": {
                        "width": 128,
                        "height": 128
                    }
                },
                "large": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/6/lm/yu/gqLmyuzMeoNfvBq-400x400-noPad.jpg?1464387230",
                    "processing": False,
                    "size": {
                        "width": 400,
                        "height": 400
                    }
                },
                "xlarge": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/6/lm/yu/gqLmyuzMeoNfvBq-800x800-noPad.jpg?1464387230",
                    "processing": False,
                    "size": {
                        "width": 800,
                        "height": 800
                    }
                }
            }
        }
    },
    {
        "id": 750547,
        "display_name": "Mitch McConnell",
        "email": None,
        "type": "Politician",
        "slug": "mitch-mcconnell",
        "description": "Addison Mitchell \"Mitch\" McConnell, Jr. is the senior United States Senator from Kentucky. A member of the Republican Party, he has been the Minority Leader of the Senate since January 3, 2007.",
        "publicly_visible": True,
        "verified_at": "2014-04-11T14:16:52Z",
        "summary": "US Senate - Kentucky",
        "locale": "en-US",
        "confirmed_at": "2013-12-12T22:31:32Z",
        "is_person": True,
        "member_of": {},
        "additional_data": {
            "state": "KY",
            "title": "Senator",
            "district": None,
            "active": True
        },
        "photo": {
            "id": 16998801,
            "url": "photos/3/gj/uv/LMgJuvNDBCnuHhF-fullsize.jpg",
            "sizes": {
                "small": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/3/gj/uv/LMgJuvNDBCnuHhF-48x48-noPad.jpg?1423788358",
                    "processing": False,
                    "size": {
                        "width": 48,
                        "height": 48
                    }
                },
                "medium": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/3/gj/uv/LMgJuvNDBCnuHhF-128x128-noPad.jpg?1423788358",
                    "processing": False,
                    "size": {
                        "width": 128,
                        "height": 128
                    }
                },
                "large": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/3/gj/uv/LMgJuvNDBCnuHhF-400x400-noPad.jpg?1423788359",
                    "processing": False,
                    "size": {
                        "width": 400,
                        "height": 400
                    }
                },
                "xlarge": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/3/gj/uv/LMgJuvNDBCnuHhF-800x800-noPad.jpg?1453772908",
                    "processing": False,
                    "size": {
                        "width": 800,
                        "height": 800
                    }
                }
            }
        }
    },
    {
        "id": 761125,
        "display_name": "Kirsten Gillibrand",
        "email": None,
        "type": "Politician",
        "slug": "kirsten-gillibrand",
        "description": "Kirsten Gillibrand is an American politician and the junior United States Senator from New York, in office since 2009. ",
        "publicly_visible": True,
        "verified_at": "2014-02-12T19:53:24Z",
        "summary": "US Senate - New York",
        "locale": "en-US",
        "confirmed_at": "2013-12-12T22:36:20Z",
        "is_person": True,
        "member_of": {},
        "additional_data": {
            "state": "NY",
            "title": "Senator",
            "district": None,
            "active": True
        },
        "photo": {
            "id": 15007365,
            "url": "photos/7/hu/wt/xkhuwtrRdJVXccL-fullsize.jpg",
            "sizes": {
                "small": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/hu/wt/xkhuwtrRdJVXccL-48x48-noPad.jpg?1423847201",
                    "processing": False,
                    "size": {
                        "width": 48,
                        "height": 48
                    }
                },
                "medium": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/hu/wt/xkhuwtrRdJVXccL-128x128-noPad.jpg?1423847201",
                    "processing": False,
                    "size": {
                        "width": 128,
                        "height": 128
                    }
                },
                "large": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/hu/wt/xkhuwtrRdJVXccL-400x400-noPad.jpg?1423847201",
                    "processing": False,
                    "size": {
                        "width": 400,
                        "height": 400
                    }
                },
                "xlarge": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/hu/wt/xkhuwtrRdJVXccL-800x800-noPad.jpg?1453772906",
                    "processing": False,
                    "size": {
                        "width": 800,
                        "height": 800
                    }
                }
            }
        }
    },
    {
        "id": 808912,
        "display_name": "U.S. House of Representatives",
        "email": None,
        "type": "Group",
        "slug": "u-s-house-of-representatives",
        "description": None,
        "publicly_visible": True,
        "verified_at": None,
        "summary": None,
        "locale": "en-US",
        "confirmed_at": "2013-12-18T21:46:55Z",
        "is_person": False,
        "member_of": {},
        "additional_data": {
            "type": "UsFederalHouse"
        },
        "photo": {
            "id": 14361706,
            "url": "photos/4/lw/he/QglWHEjPcxVGjHr-fullsize.jpg",
            "sizes": {
                "small": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/4/lw/he/QglWHEjPcxVGjHr-48x48-noPad.jpg?1423788112",
                    "processing": False,
                    "size": {
                        "width": 48,
                        "height": 48
                    }
                },
                "medium": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/4/lw/he/QglWHEjPcxVGjHr-128x128-noPad.jpg?1423788111",
                    "processing": False,
                    "size": {
                        "width": 128,
                        "height": 128
                    }
                },
                "large": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/4/lw/he/QglWHEjPcxVGjHr-400x400-noPad.jpg?1423788112",
                    "processing": False,
                    "size": {
                        "width": 400,
                        "height": 400
                    }
                },
                "xlarge": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/4/lw/he/QglWHEjPcxVGjHr-800x800-noPad.jpg?1453772985",
                    "processing": False,
                    "size": {
                        "width": 800,
                        "height": 800
                    }
                }
            }
        }
    },
    {
        "id": 808913,
        "display_name": "U.S. Senate",
        "email": None,
        "type": "Group",
        "slug": "u-s-senate",
        "description": None,
        "publicly_visible": True,
        "verified_at": None,
        "summary": None,
        "locale": "en-US",
        "confirmed_at": "2013-12-18T21:47:07Z",
        "is_person": False,
        "member_of": {},
        "additional_data": {
            "type": "UsFederalSenate"
        },
        "photo": {
            "id": 14361607,
            "url": "photos/8/hc/rk/nDHcrkPwmdaeaBD-fullsize.jpg",
            "sizes": {
                "small": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/8/hc/rk/nDHcrkPwmdaeaBD-48x48-noPad.jpg?1423799280",
                    "processing": False,
                    "size": {
                        "width": 48,
                        "height": 48
                    }
                },
                "medium": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/8/hc/rk/nDHcrkPwmdaeaBD-128x128-noPad.jpg?1423799281",
                    "processing": False,
                    "size": {
                        "width": 128,
                        "height": 128
                    }
                },
                "large": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/8/hc/rk/nDHcrkPwmdaeaBD-400x400-noPad.jpg?1423799280",
                    "processing": False,
                    "size": {
                        "width": 400,
                        "height": 400
                    }
                },
                "xlarge": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/8/hc/rk/nDHcrkPwmdaeaBD-800x800-noPad.jpg?1453773058",
                    "processing": False,
                    "size": {
                        "width": 800,
                        "height": 800
                    }
                }
            }
        }
    },
    {
        "id": 750546,
        "display_name": "Heidi Heitkamp",
        "email": None,
        "type": "Politician",
        "slug": "mitch-mcconnell",
        "description": "",
        "publicly_visible": True,
        "verified_at": "2014-04-11T14:16:52Z",
        "summary": "US Senate - Kentucky",
        "locale": "en-US",
        "confirmed_at": "2013-12-12T22:31:32Z",
        "is_person": True,
        "member_of": {},
        "additional_data": {
            "state": "ND",
            "title": "Senator",
            "district": None,
            "active": True
        },
        "photo": {
            "id": 16998801,
            "url": "photos/3/gj/uv/LMgJuvNDBCnuHhF-fullsize.jpg",
            "sizes": {
                "small": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/3/gj/uv/LMgJuvNDBCnuHhF-48x48-noPad.jpg?1423788358",
                    "processing": False,
                    "size": {
                        "width": 48,
                        "height": 48
                    }
                },
                "medium": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/3/gj/uv/LMgJuvNDBCnuHhF-128x128-noPad.jpg?1423788358",
                    "processing": False,
                    "size": {
                        "width": 128,
                        "height": 128
                    }
                },
                "large": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/3/gj/uv/LMgJuvNDBCnuHhF-400x400-noPad.jpg?1423788359",
                    "processing": False,
                    "size": {
                        "width": 400,
                        "height": 400
                    }
                },
                "xlarge": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/3/gj/uv/LMgJuvNDBCnuHhF-800x800-noPad.jpg?1453772908",
                    "processing": False,
                    "size": {
                        "width": 800,
                        "height": 800
                    }
                }
            }
        }
    },
    {
        "id": 750149,
        "display_name": "James Morris",
        "email": None,
        "type": "Politician",
        "slug": "mike-honda",
        "description": "Mike Honda proudly represents California 17th Congressional District in the U.S. House of Representatives. His district includes Silicon Valley, the birthplace of innovation and the national leader in high-tech development.",
        "publicly_visible": True,
        "verified_at": "2013-10-21T21:41:13Z",
        "summary": "US House of Representatives - California-17",
        "locale": "en-US",
        "confirmed_at": "2013-11-14T23:16:31Z",
        "is_person": True,
        "member_of": {},
        "additional_data": {
            "state": "LA",
            "title": "Representative",
            "district": "17",
            "active": True
        },
        "photo": {
            "id": 11499763,
            "url": "photos/7/su/yh/DYSuYHtdzhjvHqa-fullsize.jpg",
            "sizes": {
                "small": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/su/yh/DYSuYHtdzhjvHqa-48x48-noPad.jpg?1423780512",
                    "processing": False,
                    "size": {
                        "width": 48,
                        "height": 48
                    }
                },
                "medium": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/su/yh/DYSuYHtdzhjvHqa-128x128-noPad.jpg?1423780513",
                    "processing": False,
                    "size": {
                        "width": 128,
                        "height": 128
                    }
                },
                "large": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/su/yh/DYSuYHtdzhjvHqa-400x400-noPad.jpg?1423780514",
                    "processing": False,
                    "size": {
                        "width": 400,
                        "height": 400
                    }
                },
                "xlarge": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/su/yh/DYSuYHtdzhjvHqa-800x800-noPad.jpg?1453772907",
                    "processing": False,
                    "size": {
                        "width": 800,
                        "height": 800
                    }
                }
            }
        }
    },
    {
        "id": 750149,
        "display_name": "Barack Obama",
        "email": None,
        "type": "Politician",
        "slug": "mike-honda",
        "description": "Mike Honda proudly represents California 17th Congressional District in the U.S. House of Representatives. His district includes Silicon Valley, the birthplace of innovation and the national leader in high-tech development.",
        "publicly_visible": True,
        "verified_at": "2013-10-21T21:41:13Z",
        "summary": "US House of Representatives - California-17",
        "locale": "en-US",
        "confirmed_at": "2013-11-14T23:16:31Z",
        "is_person": True,
        "member_of": {},
        "additional_data": {
            "state": None,
            "title": "President",
            "district": "17",
            "active": True
        },
        "photo": {
            "id": 11499763,
            "url": "photos/7/su/yh/DYSuYHtdzhjvHqa-fullsize.jpg",
            "sizes": {
                "small": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/su/yh/DYSuYHtdzhjvHqa-48x48-noPad.jpg?1423780512",
                    "processing": False,
                    "size": {
                        "width": 48,
                        "height": 48
                    }
                },
                "medium": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/su/yh/DYSuYHtdzhjvHqa-128x128-noPad.jpg?1423780513",
                    "processing": False,
                    "size": {
                        "width": 128,
                        "height": 128
                    }
                },
                "large": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/su/yh/DYSuYHtdzhjvHqa-400x400-noPad.jpg?1423780514",
                    "processing": False,
                    "size": {
                        "width": 400,
                        "height": 400
                    }
                },
                "xlarge": {
                    "url": "//d22r54gnmuhwmk.cloudfront.net/photos/7/su/yh/DYSuYHtdzhjvHqa-800x800-noPad.jpg?1453772907",
                    "processing": False,
                    "size": {
                        "width": 800,
                        "height": 800
                    }
                }
            }
        }
    },

]

tp = TargetsProcessor(targets)

class TestTargetsProcessor(TestCase):



    def test_get_target_stats(self):
        pass

    def test_get_party(self):
        self.assertEqual(tp.get_party(targets[2], 2014), "R")
        self.assertEqual(tp.get_party(targets[0], 2014), "D")
        self.assertEqual(tp.get_party(targets[1], 2014), "R")
        self.assertEqual(tp.get_party(targets[3], 2014), "D")
        self.assertEqual(tp.get_party(targets[6], 2014), "D")

    def test_get_obama(self):
        self.assertEqual(tp.get_party(targets[8], 2014), "D")

    def test_get_past_responses(self):
        self.assertEqual(tp.get_past_responses(targets[2]), 1)

