# Create a  book class with the following attributes:
# •title
# •author
# •list of reviews
# And add methods to:
# •add a new review
# •count reviews
# •display all reviews

class Book:
    def __init__(self):
        self.title = ""
        self.author = ""
        self.reviews = []

    def add_review(self, review):
        self.reviews.append(review)

    def count_reviews(self):
        return len(self.reviews)
    
    def display_reviews(self):
        for review in self.reviews:
            print(review)
# Example usage:
book = Book()  
book.title = "The Great Gatsby"
book.author = "F. Scott Fitzgerald"    
book.add_review("A fascinating exploration of the American Dream.")
book.add_review("Beautifully written with complex characters.")
print(f"Total Reviews: {book.count_reviews()}")
print("Reviews:")
book.display_reviews()
