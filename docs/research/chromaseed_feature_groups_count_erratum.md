# FG protocol count correction

2026-09-13, before independent audit; primary run already completed. This corrects an arithmetic total only. Preserve the original frozen protocol and source lock f6a8ac340650ae5d62004f59916ba720b702e6dc38f1fedf6928f5797b53f4cf.

The protocol's 5,441,700 final row predictions mistakenly uses the 1,700 **fit** rows across roles. The registered held roles actually contain 232 + 643 + 323 = 1,198 query rows. With97 cases per role and33 transforms, the correct full count is:

`97 * (232 + 643 + 323) * 33 = 3,834,798`.

The288 nonconstant independent refits cover96 *1,198 *33 =3,795,264 predictions; constants account for39,534. Inner OOF remains289 *1,700 =491,300 predictions. No rows, models, transforms, selections, thresholds or fit data have been added or removed. The primary records and saved prediction shapes already implement the registered role definitions. This erratum is bound into the independent audit and reported explicitly.

