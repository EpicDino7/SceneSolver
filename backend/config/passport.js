import passport from "passport";
import { Strategy as GoogleStrategy } from "passport-google-oauth20";
import dotenv from "dotenv";
import Guser from "../models/Guser.js";

dotenv.config();

passport.serializeUser((user, done) => {
  done(null, user.id);
});

passport.deserializeUser(async (id, done) => {
  try {
    const user = await Guser.findById(id);
    done(null, user);
  } catch (error) {
    done(error, null);
  }
});

// Determine the correct callback URL based on environment
const getCallbackURL = () => {
  if (process.env.NODE_ENV === "production") {
    return "https://scenesolver-backend-2wb1.onrender.com/api/auth/google/callback";
  }
  return "/api/auth/google/callback"; // Relative URL for development
};

passport.use(
  new GoogleStrategy(
    {
      clientID: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      callbackURL: getCallbackURL(),
    },
    async (accessToken, refreshToken, profile, done) => {
      try {
        let user = await Guser.findOne({ googleId: profile.id });

        if (!user) {
          user = await new Guser({
            googleId: profile.id,
            email: profile.emails[0].value,
            displayName: profile.displayName,
            avatar: profile.photos[0].value,
          }).save();
        }

        done(null, user);
      } catch (error) {
        done(error, null);
      }
    }
  )
);

export default passport;
