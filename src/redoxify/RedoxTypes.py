# this file is supposed to be imported by user into his model config

from typing import List, Dict
from attrs import define, field
from typing import Union

@define(eq=True)    # must be comparable
class DataKey:
    main_key : str = field()
    sub_key : Union[str, None] = field(default=None)
    
class ImageEncoding:
    jpeg : str = 'jpg'
    png : str = 'png'
    raw : str = 'raw'
    
    @classmethod
    def get_all_encodings(cls) -> List[str]:
        return [cls.jpeg, cls.png, cls.raw]