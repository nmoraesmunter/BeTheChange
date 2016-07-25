from unittest import TestCase
from data_collector import DataCollector
import codecs


class TestDataCollector(TestCase):



    def test_webscrape_user(self):
        petition_html = codecs.open("test_petition.html", 'r').read()
        user_html = codecs.open("test_user.html", 'r').read()

        new_fields = DataCollector.webscrape(None, petition_html, user_html, "user")

        self.assertEqual(new_fields["creator_type"], "user")
        self.assertFalse(new_fields["creator_has_website"])
        self.assertEqual(new_fields["creator_city"], "dallas")
        self.assertEqual(new_fields["creator_first_name"], "Clenesha")
        self.assertEqual(new_fields["creator_last_name"], "Garland")
        self.assertEqual(new_fields["creator_country"], "US")
        self.assertEqual(new_fields["creator_description"], None)
        self.assertEqual(new_fields["creator_display_name"], "Clenesha Garland")
        self.assertEqual(new_fields["creator_locale"], "en-US")
        self.assertTrue(new_fields["creator_has_photo"])
        self.assertEqual(new_fields["creator_state"], "TX")
        self.assertEqual(new_fields["creator_fb_permissions"], 0)
        self.assertTrue(new_fields["creator_has_slug"])

        self.assertEqual(new_fields["num_past_petitions"], 1)
        self.assertEqual(new_fields["num_past_victories"], 1)
        self.assertEqual(new_fields["num_past_verified_victories"], 1)
        self.assertEqual(new_fields["last_past_victory_date"], '2015-12-18')
        self.assertEqual(new_fields["last_past_verified_victory_date"], '2015-12-18')


        self.assertEqual(new_fields["ask"], 'President Barack Obama: Sharanda Jones does not deserve to die in prison')
        self.assertEqual(new_fields["calculated_goal"], 300000)
        self.assertGreater(len(new_fields["description"]), 10)
        self.assertTrue(new_fields["discoverable"])
        self.assertEqual(new_fields["display_title"], 'President Barack Obama: Sharanda Jones does not deserve to die in prison')
        self.assertEqual(new_fields["displayed_signature_count"], 279890)
        self.assertFalse(new_fields["is_pledge"], False)
        self.assertEqual(new_fields["is_victory"], True)
        self.assertEqual(new_fields["is_verified_victory"], True)
        self.assertEqual(new_fields["languages"], ['en'])
        self.assertEqual(new_fields["original_locale"], 'en-US')
        self.assertAlmostEqual(new_fields["progress"], 93.2966666667)
        self.assertEqual(len(new_fields["tags"]), 9)
        self.assertEqual(new_fields["victory_date"], '2015-12-18')
        self.assertTrue(new_fields["has_video"])
        self.assertTrue(new_fields["has_photo"])
        self.assertEqual(len(new_fields["targets_detailed"]), 1)


        self.assertEqual(len(new_fields), 35)



    def test_webscrape_org(self):
        petition_html = codecs.open("test_petition.html", 'r').read()
        user_html = codecs.open("test_org.html", 'r').read()

        new_fields = DataCollector.webscrape(None, petition_html, user_html, "org")

        self.assertEqual(new_fields["creator_type"], "org")
        self.assertTrue(new_fields["creator_has_website"])
        self.assertEqual(new_fields["creator_city"], "Washington")
        self.assertTrue(new_fields["creator_has_photo"])
        self.assertEqual(new_fields["creator_country"], "US")
        self.assertEqual(new_fields["creator_state"], "DC")
        self.assertTrue(new_fields["creator_has_slug"])


        self.assertTrue(new_fields["creator_has_address"])
        self.assertFalse(new_fields["creator_has_contact_email"])
        self.assertFalse(new_fields["creator_has_fb_page"], None)
        self.assertTrue(new_fields["creator_mission"])
        self.assertEqual(new_fields["creator_org_name"], "International Labor Rights Forum")
        self.assertEqual(new_fields["creator_tax_country_code"], None)
        self.assertEqual(new_fields["creator_tax_state_code"], None)
        self.assertEqual(new_fields["creator_zipcode"], "20006")
        self.assertEqual(new_fields["creator_postal_code"], "20006")

        self.assertFalse(new_fields["creator_has_twitter"])
        self.assertFalse(new_fields["creator_has_verified_req"])
        self.assertFalse(new_fields["creator_has_verified_by"])
        self.assertFalse(new_fields["creator_has_verified_at"])
        self.assertFalse(new_fields["creator_has_video"])


        self.assertEqual(new_fields["num_past_petitions"], 5)
        self.assertEqual(new_fields["num_past_victories"], 5)
        self.assertEqual(new_fields["num_past_verified_victories"], 5)
        self.assertEqual(new_fields["last_past_victory_date"], '2011-10-18')
        self.assertEqual(new_fields["last_past_verified_victory_date"], '2011-10-18')

        self.assertEqual(len(new_fields), 43)






