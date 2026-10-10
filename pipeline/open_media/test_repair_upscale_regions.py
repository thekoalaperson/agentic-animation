import copy
import unittest

import numpy as np

from repair_upscale_regions import masks_at, validate_recipe


class RegionalRepairTests(unittest.TestCase):
    def setUp(self):
        self.recipe = {"approved": True, "kind": "source-preserving-background-repair",
                       "chunkStartFrame": 100, "chunkFrames": 20, "globalFrameRange": [104,116],
                       "regions": [{"outerBox": [20,20,100,100], "featherPixels": 8}]}

    def test_static_and_exclusive_interval(self):
        validate_recipe(self.recipe)
        self.assertFalse(list(masks_at(self.recipe,103)))
        self.assertFalse(list(masks_at(self.recipe,116)))
        alpha = list(masks_at(self.recipe,104))[0][-1]
        self.assertEqual(float(alpha[0,40,0]), 0)
        self.assertEqual(float(alpha[40,40,0]), 1)

    def test_moving_box_and_temporal_fade(self):
        region = self.recipe["regions"][0]
        del region["outerBox"]
        region["outerBoxKeyframes"] = [{"frame":104,"outerBox":[20,20,100,100]}, {"frame":116,"outerBox":[32,32,112,112]}]
        self.recipe["fadeOutFrames"] = 4
        validate_recipe(self.recipe)
        self.assertEqual(list(masks_at(self.recipe,110))[0][:4], (26,26,106,106))
        peaks = [float(list(masks_at(self.recipe,f))[0][-1].max()) for f in range(112,116)]
        np.testing.assert_allclose(peaks,[1,.85355339,.5,.14644661],atol=1e-6)

    def test_independent_regions_at_cut(self):
        self.recipe["regions"][0]["globalFrameRange"] = [104,110]
        second = copy.deepcopy(self.recipe["regions"][0])
        second["globalFrameRange"] = [110,116]
        second["outerBox"] = [120,120,200,200]
        self.recipe["regions"].append(second)
        validate_recipe(self.recipe)
        self.assertEqual(list(masks_at(self.recipe,109))[0][0],20)
        self.assertEqual(list(masks_at(self.recipe,110))[0][0],120)

    def test_full_source_at_picture_boundary_only(self):
        self.recipe['kind']='source-preserving-regional-detail-repair'
        region=self.recipe['regions'][0]
        region.update(outerBox=[100,980,200,1080],featherPixels=12,featherFrameEdges=False)
        validate_recipe(self.recipe)
        alpha=list(masks_at(self.recipe,104))[0][-1]
        self.assertEqual(float(alpha[-1,50,0]),1)
        self.assertEqual(float(alpha[0,50,0]),0)
        self.assertEqual(float(alpha[-1,0,0]),0)
        region['featherFrameEdges']=True
        self.assertEqual(float(list(masks_at(self.recipe,104))[0][-1][-1,50,0]),0)

    def test_bad_approvals_geometry_and_unknown_fields(self):
        for mutation in [lambda r:r.update(approved=False), lambda r:r.update(globalFrameRange=[90,116]),
                         lambda r:r.update(fadeOutFrames=50), lambda r:r.update(fadeFrames=4),
                         lambda r:r["regions"][0].update(outerBox=[0,0,10,10]),
                         lambda r:r["regions"][0].update(fade=4)]:
            r = copy.deepcopy(self.recipe)
            mutation(r)
            with self.assertRaises(ValueError):
                validate_recipe(r)


if __name__ == "__main__":
    unittest.main()
