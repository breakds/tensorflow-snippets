class Dictionary:
  """Utility class maintaining maps between character and their indices."""

  def __init__(self, paths, ascii_only=True):
    """Create the dictionary from characters in the given files."""
    # When ascii_only is set to True, all non-ascii characters will be
    # mapped to unknown.
    
    # The list of characters
    self._characters = []

    # Hashmap that maps character to its corresponding indices.
    self._indices = {}
    
    for path in paths:
      with open(path, 'r') as f:
        while True:
          character = f.read(1)
          if not character:
            break
          elif character not in self._indices:
            if ascii_only and ord(character) > 255:
              continue
            self._indices[character] = len(self._characters)
            self._characters.append(character)

  def character(self, index):
    if index >= len(self._characters):
      return ' '
    return self._characters[index]

  def index(self, character):
    try:
      return self._indices[character]
    except KeyError:
      return len(self._character)

  def Translate(self, indices):
    return ''.join([self.character(index) for index in indices])

  @property
  def size(self):
    # Plus 1 for the "unknow" character.
    return len(self._characters) + 1

  def __str__(self):
    return ('[' + str(len(self._characters)) + '] (' +
            ' '.join(self._characters) +')')
